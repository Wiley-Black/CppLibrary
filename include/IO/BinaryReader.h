/*	BinaryReader.h
	Copyright (C) 2024 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBBinaryReader_h__
#define __WBBinaryReader_h__

#include "wbFoundation.h"
#include "../IO/Streams.h"
#include "../Memory Management/Allocation.h"
#include "../IO/FileStream.h"

namespace wb
{
	namespace io
	{
		class BinaryReader
		{
		protected:
			bool						m_LittleEndianStream;
			bool						m_SwapEndian;
			memory::r_ptr<Stream>		m_pStream;					

			void ReadEndian(void* pData, UInt64 length);

		public:
			BinaryReader(memory::r_ptr<Stream>&& rpStream, bool little_endian_stream = true)
				: 												
				m_LittleEndianStream(little_endian_stream),
				m_pStream(std::move(rpStream))
			{ 
				m_SwapEndian = (IsLittleEndian() != m_LittleEndianStream);
			}

			BinaryReader(const Path& path, bool little_endian_stream = true)
				:				
				m_LittleEndianStream(little_endian_stream),
				m_pStream(memory::r_ptr<Stream>::responsible(new FileStream(
					path.to_osstring(), FileMode::Open, 
					FileAccess::Read, FileShare::Read)))
			{ 
				m_SwapEndian = (IsLittleEndian() != m_LittleEndianStream);
			}

			/// <summary>
			/// Reads 'length' bytes directly from the stream.  No endian swapping is performed.
			/// </summary>			
			/// <returns>The number of bytes read.  May be less than length if the end of stream
			/// was reached.</returns>
			Int64 Read(void* pBuffer, UInt64 length);

			byte ReadByte();
			char ReadChar();
			Int16 ReadInt16() { Int16 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			Int32 ReadInt32() { Int32 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			Int64 ReadInt64() { Int64 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			UInt16 ReadUInt16() { UInt16 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			UInt32 ReadUInt32() { UInt32 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			UInt64 ReadUInt64() { UInt64 ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			float ReadSingle() { float ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			double ReadDouble() { double ret; ReadEndian(&ret, sizeof(ret)); return ret; }
			void Seek(Int64 offset, wb::io::SeekOrigin origin);
			Int64 GetPosition();
			void Flush();
			Int64 GetLength() { return m_pStream->GetLength(); }

			void SetEndianness(bool little_endian_stream);
		};

		/** Implementation **/

		inline void BinaryReader::SetEndianness(bool little_endian_stream)
		{
			m_SwapEndian = (IsLittleEndian() != m_LittleEndianStream);
		}

		inline Int64 BinaryReader::Read(void* pBuffer, UInt64 length)
		{
			return m_pStream->Read(pBuffer, length);
		}

		inline byte BinaryReader::ReadByte() 
		{ 
			int ret = m_pStream->ReadByte(); 
			if (ret < 0)
				throw IOException("End of stream reached while retrieving next byte from stream.");
			return (byte)ret;
		}

		inline char BinaryReader::ReadChar() { return (char)ReadByte(); }

		inline void BinaryReader::ReadEndian(void* pData, UInt64 length)
		{
			if (length == 0) return;
			Int64 nBytesRead = m_pStream->Read(pData, length);
			if (nBytesRead != length)
				throw IOException("Unable to read requested length from stream.");
			if (length == 1)
				return;
			if (m_SwapEndian)
			{
				if (length & 1)
					throw ArgumentException("Cannot swap endianness of a value that is not an even number of bytes in length.");
				byte* pFwdData = ((byte*)pData);
				byte* pRevData = ((byte*)pData) + (length - 1);
				for (UInt64 ii = 0; ii < (length >> 1); ii++, pRevData--, pFwdData++)
				{
					byte ch = *pFwdData;
					*pFwdData = *pRevData;
					*pRevData = ch;
				}
			}			
		}

		inline void BinaryReader::Seek(Int64 offset, wb::io::SeekOrigin origin)
		{
			m_pStream->Seek(offset, origin);
		}

		inline Int64 BinaryReader::GetPosition()
		{
			return m_pStream->GetPosition();
		}

		inline void BinaryReader::Flush()
		{
			m_pStream->Flush();
		}
	}
}

#endif	// __WBBinaryReader_h__

//	End of BinaryReader.h

