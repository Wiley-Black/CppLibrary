/////////
//	DeflateStream_InternalImpl.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBDeflateStream_Internal_h__
#define __WBDeflateStream_Internal_h__

#ifdef DeflateStream_InternImpl

/** Table of contents **/

namespace wb {
	namespace io {
		namespace compression {
			class DeflateStream;
		}
	}
}

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../Foundation/STL/Memory.h"
#include "../../Memory Management/HistoryRingBuffer.h"
#include "../BitStream.h"
#include "HuffmanTable.h"

/** Content **/

namespace wb {
	namespace io {
		namespace compression
		{
			using namespace wb::memory;			// for r_ptr.

			enum_class_start(CompressionMode,int)
			{
				Compress,
				Decompress
			}
			enum_class_end(CompressionMode);
			
			class DeflateStream : public Stream
			{				
				LSBBitStream	m_BitStream;

				/// <summary>
				/// The DEFLATE compression technique supports distance codes whereby
				/// an instruction tells the decoder to look back in history and repeat
				/// something that was previously decompressed.  The HistoryRingBuffer
				/// facilitates this.  History in DEFLATE can cross block boundaries,
				/// but the largest distance code allowed is 32768.
				/// </summary>
				enum { MaxHistory = 32768 };
				wb::memory::HistoryRingBuffer<byte>		m_History;

				enum { MAXBITS = 15 };              /* maximum bits in a code */        

				enum_class_start(BlockType,int)
				{
					None,
					Noncompressed,
					FixedHuffman,
					DynamicHuffman
				}
				enum_class_end(BlockType);

				BlockType		m_BlockType;
				bool			m_bFinalBlock;
				bool			m_EOS;			// End of output has been reached.

				enum { BlockSize = 65535 };
				MemoryStream	m_OutputBuffer;
				
				void StartNextBlock();				

				/** Non-compressed blocks **/
				int		m_NCBlockRemaining;
				void	StartNoncompressedBlock();
				byte	ReadNCByte();
				Int64	ReadNCByte(void *pBuffer, Int64 nLength);

				/** Compressed blocks **/

				r_ptr<HuffmanTable>		m_pCurrentLengthCodes;
				r_ptr<HuffmanTable>		m_pCurrentDistanceCodes;

				/// <summary>
				/// When more than one byte is decoded in a single operation, we have to store the
				/// extra bytes somewhere (assuming we are reading a byte at a time).  We already
				/// have to maintain 'History' in order to facilitate backward string copies that
				/// are a cornerstone of deflate's compression mechanism.  When QueuedHistory is
				/// greater than zero, it indicates a number of bytes which are stored in History 
				/// but have not yet been returned from the DeflateStream.  QueuedHistory should 
				/// never exceed 258 bytes based on deflate rules.
				/// </summary>
				int m_QueuedHistory;

				static uint RepeatLengthBase[];				
				static uint BackwardDistanceBase[];
				static int RepeatLengthExtraBits[];					
				static int BackwardDistanceExtraBits[];

				int ReadCompressedByte();

				/** Fixed-Length Huffman code blocks **/

				class FixedLengthHuffmanTable : public HuffmanTable
				{
				public:
					FixedLengthHuffmanTable();
				};

				class FixedDistanceHuffmanTable : public HuffmanTable
				{
				public:
					FixedDistanceHuffmanTable();
				};

				static FixedLengthHuffmanTable g_FixedLengthCodes;
				static FixedDistanceHuffmanTable g_FixedDistanceCodes;
				void StartFixedHuffmanBlock();

				/** Dynamic Huffman code blocks **/

				enum { HeaderCLSequenceLength = 19 };
				static const int HeaderCLSequence[HeaderCLSequenceLength];
				HuffmanTable* ReadHeaderCodes(int HCLEN);
				void StartDynamicHuffmanBlock();

			public:

				enum_class_start(Flags,uint)
				{
					None		= 0x0,
					SkipHeaders = 0x1
				}
				enum_class_end(Flags);

				/** Pass a pointer to the underlying stream to the constructor for the DeflateStream to take responsibility for deleting the
					object.  Pass a reference to avoid transferring responsibility for the object. **/
				DeflateStream(r_ptr<Stream>&& UnderlyingStream, CompressionMode mode, Flags flags);				
				~DeflateStream();

				bool CanRead() override { return true; }
				bool CanWrite() override { return false; }
				bool CanSeek() override { return false; }
				
				int ReadByte() override;
				Int64 Read(void *pBuffer, Int64 nLength) override;

				void Close() override { if (m_BitStream.m_pUnderlying.IsResponsible()) m_BitStream.m_pUnderlying->Close(); }
				void Flush() override { m_BitStream.m_pUnderlying->Flush(); }
			};

			/** Implementation **/

			inline DeflateStream::DeflateStream(r_ptr<Stream>&& UnderlyingStream, CompressionMode mode, Flags flags)
				: m_BitStream(std::move(UnderlyingStream)),
				m_History(MaxHistory)
			{
				if (mode != CompressionMode::Decompress) throw NotSupportedException();
				m_BlockType = BlockType::None;
				m_bFinalBlock = false;
				m_EOS = false;

				m_NCBlockRemaining = 0;
				m_QueuedHistory = 0;
			}		

			inline DeflateStream::~DeflateStream()
			{
			}						

			inline Int64 DeflateStream::Read(void *pBuffer, Int64 nLength) 
			{
				byte* pbBuffer = (byte*)pBuffer;
				Int64 nRead;
				for (nRead = 0; nRead < nLength; )
				{
					switch (m_BlockType)
					{
					case BlockType::Noncompressed:
						{
							Int64 nBlockRead = ReadNCByte(pbBuffer + nRead, nLength - nRead);                            
							nRead += nBlockRead;
							continue;
						}

					default:
						int nByte = ReadByte();
						if (nByte < 0) return nRead;
						pbBuffer[nRead++] = (byte)nByte;
						continue;
					}
				}
				return nRead;
			}

			inline int DeflateStream::ReadByte()
			{
				int ret;
				for(;;)
				{
					switch (m_BlockType)
					{
					case BlockType::None: 
						if (m_EOS) return -1;
						StartNextBlock();
						continue;

					case BlockType::Noncompressed:                        
						ret = ReadNCByte();     // ReadNCByte() returns only a byte, EOS is unexpected.                        
						return ret;

					case BlockType::FixedHuffman:
					case BlockType::DynamicHuffman:
						ret = ReadCompressedByte();
						if (ret < 0) continue;                        
						return ret;

					default: throw NotSupportedException("Illegal block type.");
					}
				}
			}

			inline void DeflateStream::StartNextBlock()
			{
				m_bFinalBlock = (m_BitStream.ReadBits(1) != 0);
				uint BTYPE = m_BitStream.ReadBits(2);

				switch (BTYPE)
				{
					case 0:
						m_BlockType = BlockType::Noncompressed;
						StartNoncompressedBlock();
						break;

					case 1:
						m_BlockType = BlockType::FixedHuffman;
						StartFixedHuffmanBlock();
						break;

					case 2:
						m_BlockType = BlockType::DynamicHuffman;
						StartDynamicHuffmanBlock();
						break;

					default: throw FormatException("Unsupported block type within DEFLATE stream.");
				}
			}

			/** Implementation - DeflateStream - Non-Compressed Blocks **/						

			inline void DeflateStream::StartNoncompressedBlock()
			{
				m_BitStream.FlushCurrentByte();
				
				m_NCBlockRemaining = ((int)m_BitStream.ReadByte()) | ((int)m_BitStream.ReadByte() << 8);
				int ComplBlockLength = ((int)m_BitStream.ReadByte()) | ((int)m_BitStream.ReadByte() << 8);
				if (m_NCBlockRemaining != (~ComplBlockLength & 0xFFFF))
					throw FormatException("Corrupt or invalid block length in non-compressed block.");				
			}

			inline byte DeflateStream::ReadNCByte()
			{
				if (m_NCBlockRemaining > 1)
				{
					m_NCBlockRemaining--;
					byte ret = m_BitStream.ReadByte();
					m_History.Add(ret); 
					return ret;
				}

				if (m_NCBlockRemaining == 1)
				{
					byte ret = m_BitStream.ReadByte();
					m_History.Add(ret);

					m_NCBlockRemaining = 0;
					m_BlockType = BlockType::None;
					if (m_bFinalBlock) m_EOS = true;
                
					return ret;
				}

				throw NotSupportedException();
			}

			inline Int64 DeflateStream::ReadNCByte(void *pBuffer, Int64 nLength)
			{				
				byte* pbBuffer = (byte*) pBuffer;

				if ((Int64)m_NCBlockRemaining > nLength)
				{
					for (Int64 ii = 0; ii < nLength; ii++) pbBuffer[ii] = m_BitStream.ReadByte();
					m_History.Add(pbBuffer, (int)nLength);
					m_NCBlockRemaining -= (int)nLength;
					return nLength;
				}

				Int64 nBytes;
				if (m_NCBlockRemaining == nLength) nBytes = nLength;
				else nBytes = m_NCBlockRemaining;
            
				for (Int64 ii = 0; ii < nBytes; ii++) pbBuffer[ii] = m_BitStream.ReadByte();
				while (nBytes > Int32_MaxValue) { m_History.Add(pbBuffer, Int32_MaxValue); pbBuffer += Int32_MaxValue; }
				m_History.Add(pbBuffer, (int)nBytes);
				m_NCBlockRemaining = 0;
				m_BlockType = BlockType::None;
				if (m_bFinalBlock) m_EOS = true;
				return nBytes;
			}

			/** Implementation - DeflateStream - Fixed-Length Huffman Code Setup **/			
			
			inline DeflateStream::FixedLengthHuffmanTable::FixedLengthHuffmanTable()
			{
				unique_ptr<uint[]> CodeLength(new uint [288]);

                int symbol = 0;
                for (; symbol <= 143; symbol++) CodeLength[symbol] = 8;
                for (; symbol <= 255; symbol++) CodeLength[symbol] = 9;
                for (; symbol <= 279; symbol++) CodeLength[symbol] = 7;
                for (; symbol <= 287; symbol++) CodeLength[symbol] = 8;
                Init(CodeLength.get(), 288);
			}			

			inline DeflateStream::FixedDistanceHuffmanTable::FixedDistanceHuffmanTable()
			{
				unique_ptr<uint[]> CodeLength(new uint [32]);

                for (int symbol = 0; symbol < 32; symbol++) CodeLength[symbol] = 5;
                Init(CodeLength.get(), 32);
			}

			#ifdef PrimaryModule
			/*static*/ DeflateStream::FixedLengthHuffmanTable DeflateStream::g_FixedLengthCodes;
			/*static*/ DeflateStream::FixedDistanceHuffmanTable DeflateStream::g_FixedDistanceCodes;
			#endif

			inline void DeflateStream::StartFixedHuffmanBlock()
			{
				// Pass the global variables to the r_ptr objects by absolved so that the r_ptr takes
				// no responsibility over them (avoid deleting the static variables)...
				m_pCurrentLengthCodes = r_ptr<HuffmanTable>::absolved(g_FixedLengthCodes);
				m_pCurrentDistanceCodes = r_ptr<HuffmanTable>::absolved(g_FixedDistanceCodes);
			}

			/** Implementation - DeflateStream - Dynamic Huffman Code Setup **/			

			#ifdef PrimaryModule
			/*static*/ const int DeflateStream::HeaderCLSequence[HeaderCLSequenceLength] 
				= { 16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15 };
			#endif

			inline HuffmanTable* DeflateStream::ReadHeaderCodes(int HCLEN)
			{
				uint CodeLength[HeaderCLSequenceLength];
				ZeroMemory (CodeLength, sizeof(CodeLength));
				for (int ii = 0; ii < HCLEN; ii++) 
				{
					assert (HeaderCLSequence[ii] < HeaderCLSequenceLength);
					CodeLength[HeaderCLSequence[ii]] = m_BitStream.ReadBits(3);										
				}
				return new HuffmanTable(CodeLength, HeaderCLSequenceLength);
			}

			inline void DeflateStream::StartDynamicHuffmanBlock()
			{
				// The dynamic huffman block consists of a custom literal/length alphabet and a custom distance
				// alphabet.  These alphabets are applied to the block's compressed content.  The alphabets
				// themselves are stored using a Huffman encoding which we will call the 'Header' alphabet.
            
				uint HLIT = m_BitStream.ReadBits(5) + 257;        // # of Literal/Length codes - 257 (257 - 286)
				uint HDIST = m_BitStream.ReadBits(5) + 1;         // # of Distance codes - 1 (1 - 32)
				if (HLIT > 286 || HDIST > 32) throw FormatException("Compressed data corrupt: Invalid header values in dynamic block.");

				uint HCLEN = m_BitStream.ReadBits(4) + 4;         // # of Code Length codes - 4 (4 - 19)
				unique_ptr<HuffmanTable> pHeaderCodes = unique_ptr<HuffmanTable>(ReadHeaderCodes((int)HCLEN));

				/** Read dynamic literal/length and distance tables **/

				/**
					* The code length repeat codes can cross from the literal/length 
					* alphabet block to the distance alphabet block.  Thus we need
					* to read in all the code lengths for the two alphabets in one
					* operation, then we can divy them out into the two alphabets.
					*/
            
				unique_ptr<uint[]> DynCodeLength(new uint[HLIT + HDIST]);
				ZeroMemory (DynCodeLength.get(), sizeof(uint) * (HLIT + HDIST));

				uint Repeats = 0;
				uint PrevCodeLength = 0;             
				for (uint ii = 0; ii < HLIT + HDIST; )
				{
					if (Repeats > 0)
					{
						DynCodeLength[ii++] = PrevCodeLength;
						Repeats--;
						continue;
					}

					uint symbol = pHeaderCodes->Decode(m_BitStream);

					if (symbol < 16)                // 0..15: Literal value of the code length
					{
						DynCodeLength[ii++] = symbol;
						PrevCodeLength = symbol;
					}
					else if (symbol == 16)          // 16: Repeat last symbol N times, where N is 3..6
					{
						if (ii == 0) throw FormatException("Compressed stream corrupt: Header used symbol repeat on first value.");
						Repeats = m_BitStream.ReadBits(2) + 3;
					}
					else if (symbol == 17)          // 17: Repeat zero value N times, where N is 3..10
					{
						PrevCodeLength = 0;
						Repeats = m_BitStream.ReadBits(3) + 3;
					}
					else if (symbol == 18)          // 18: Repeat zero value N times, where N is 11..138
					{
						PrevCodeLength = 0;
						Repeats = m_BitStream.ReadBits(7) + 11;
					}
					else throw NotSupportedException();
				}

				/** Split the dynamic code length list into the literal/length table and the distance table **/
				unique_ptr<uint[]> LitCodeLength(new uint[HLIT]);
				unique_ptr<uint[]> DistCodeLength(new uint[HDIST]);

				for (uint ii = 0; ii < HLIT; ii++) LitCodeLength[ii] = DynCodeLength[ii];
				for (uint jj = 0, ii = HLIT; jj < HDIST; ii++, jj++) DistCodeLength[jj] = DynCodeLength[ii];

				// Verify that the end of block code (256) is present...
				if (HLIT < 256 || LitCodeLength[256] == 0) throw FormatException("Compression stream corrupt: Dynamic block did not contain a termination code.");

				// By passing the HuffmanTables as responsible, the r_ptr will take responsibility for deletion.
				m_pCurrentLengthCodes = r_ptr<HuffmanTable>::responsible(new HuffmanTable(LitCodeLength.get(), HLIT));
				m_pCurrentDistanceCodes = r_ptr<HuffmanTable>::responsible(new HuffmanTable(DistCodeLength.get(), HDIST));
			}

			/** Implementation - DeflateStream - Compressed block decoding **/			

			#ifdef PrimaryModule
			/*static*/ uint DeflateStream::RepeatLengthBase[] = 
				{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258};        
			/*static*/ uint DeflateStream::BackwardDistanceBase[] = 
				{ 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 
					6145, 8193, 12289, 16385, 24577};
			/*static*/ int DeflateStream::RepeatLengthExtraBits[] =
				{ 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0 };
			/*static*/ int DeflateStream::BackwardDistanceExtraBits[] = 
				{ 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13};
			#endif

			inline int DeflateStream::ReadCompressedByte()
			{
				if (m_QueuedHistory > 0) return m_History.GetHistory(m_QueuedHistory--);
            
				uint symbol = m_pCurrentLengthCodes->Decode(m_BitStream);
				if (symbol < 256)
				{
					// Symbols 0..255:  The symbol is the literal content.
					m_History.Add((byte)symbol);
					return (int)symbol;
				}
				else if (symbol > 256)
				{
					// Symbols 257..285:  The symbol indicates a <length, backward distance> pair.
					//                    The symbol value gives part of the length information.

					if (symbol > 285) throw FormatException("Compressed stream corrupt: An invalid lit/length symbol (greater than 285) was encountered.");
					symbol -= 257;
					uint RepeatLength = RepeatLengthBase[symbol] + m_BitStream.ReadBits(RepeatLengthExtraBits[symbol]);

					// Read distance symbol...
					symbol = m_pCurrentDistanceCodes->Decode(m_BitStream);
					uint BackwardDistance = BackwardDistanceBase[symbol] + m_BitStream.ReadBits(BackwardDistanceExtraBits[symbol]);
					if (BackwardDistance > (uint)m_History.length()) throw FormatException("Compressed stream corrupt: backward distance exceeded history.");

					// Add the repeated string to the history, but mark that part of the history is pending return to the
					// caller with QueuedHistory...
					for (uint ii = 0; ii < RepeatLength; ii++) m_History.Add((byte)m_History.GetHistory(BackwardDistance));
					m_QueuedHistory += (int)RepeatLength;
					// Return the first byte from the queued history, and remove it from the queue.
					return m_History.GetHistory(m_QueuedHistory--);
				}
				else if (symbol == 256)
				{
					// Symbol 256:  End of block
					m_BlockType = BlockType::None;
					if (m_bFinalBlock) m_EOS = true;
					return -1;          // Caller needs to check for another block, unless EOS is set.
				}

				throw NotSupportedException();
			}
		}
	}
}

#endif

#endif	// __WBDeflateStream_Internal_h__

//	End of DeflateStream_InternalImpl.h

