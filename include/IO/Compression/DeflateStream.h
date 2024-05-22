/////////
//	DeflateStream.h
//	Copyright (C) 2014 by Wiley Black
//
//	Depends on the zlib library.
////

#ifndef __WBDeflateStream_h__
#define __WBDeflateStream_h__

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

#if defined(DeflateStream_ZLibImpl) && defined(DeflateStream_InternImpl)
	#error Only one of the two DeflateStream implementations may be selected.  See Configuration.h.
#endif

#if !defined(DeflateStream_ZLibImpl) && !defined(DeflateStream_InternImpl) && !defined(No_DeflateStream)
	#error One of the two DeflateStream implementations must be selected.  See Configuration.h.
#endif

#ifdef DeflateStream_ZLibImpl

/** DeflateStream Dependencies **/

#ifdef DeflateStream_ZLibImpl
	#define ZLIB_WINAPI
	#include "zlib.h"

	#if defined(_MSC_VER)		// Microsoft Visual C++ Compiler in use	
	#pragma comment(lib, "zlibstat.lib")		
	#endif
#endif

#ifdef _DLL
	#error As presently configured, zlibstat.lib requires a build with statically linked run-time library.  See Code Generation under project options.
#endif

#include "../Streams.h"
#include "../../Memory Management/Allocation.h"
#include "../MemoryStream.h"

/** Content **/

namespace wb
{
	namespace io
	{
		namespace compression
		{
			enum class CompressionMode
			{
				Compress,
				Decompress
			};

			class DeflateStream : public Stream
			{
				r_ptr<Stream>	m_pUnderlying;
				z_stream		m_zdata;
				bool			m_EOI;			// End of input has been reached.
				bool			m_EOO;			// End of output has been reached.
				void DoThrow(int ZLibStatusCode);

				enum { BlockSize = 65535 };
				MemoryStream	m_InputBuffer;
				MemoryStream	m_OutputBuffer;
				void GenerateOutput();

			public:				

				enum class Flags : uint
				{
					None		= 0x0,
					SkipHeaders = 0x1
				};

				/** Pass a pointer to the underlying stream to the constructor for the DeflateStream to take responsibility for deleting the
					object.  Pass a reference to avoid transferring responsibility for the object. **/
				DeflateStream(Stream& UnderlyingStream, CompressionMode mode, Flags flags = Flags::None);
				DeflateStream(Stream* UnderlyingStream, CompressionMode mode, Flags flags = Flags::None);
				~DeflateStream();

				bool CanRead() override { return true; }
				bool CanWrite() override { return false; }
				bool CanSeek() override { return false; }
				
				int ReadByte() override;
				Int64 Read(void *pBuffer, Int64 nLength) override;

				void Close() override { if (m_pUnderlying.IsResponsible()) m_pUnderlying->Close(); }
				void Flush() override { m_pUnderlying->Flush(); }
			};

			/** Implementation **/

			inline DeflateStream::DeflateStream(Stream& UnderlyingStream, CompressionMode mode, Flags flags)
				: m_pUnderlying(UnderlyingStream)
			{
				if (mode != CompressionMode::Decompress) throw NotSupportedException();
				m_EOI = false;
				m_EOO = false;

				m_zdata.zalloc = nullptr;
				m_zdata.zfree = nullptr;
				m_zdata.opaque = nullptr;
				m_zdata.next_in = nullptr;
				m_zdata.avail_in = 0;
				m_zdata.next_out = nullptr;
				m_zdata.avail_out = 0;
				int window_bits = ((uint)flags & (uint)Flags::SkipHeaders) ? -MAX_WBITS : MAX_WBITS;
				int sc = inflateInit2 (&m_zdata, window_bits);
				if (sc != Z_OK) DoThrow(sc);
			}

			inline DeflateStream::DeflateStream(Stream* UnderlyingStream, CompressionMode mode, Flags flags)
				: m_pUnderlying(UnderlyingStream)
			{
				if (mode != CompressionMode::Decompress) throw NotSupportedException();
				m_EOI = false;
				m_EOO = false;

				m_zdata.zalloc = nullptr;
				m_zdata.zfree = nullptr;
				m_zdata.opaque = nullptr;
				m_zdata.next_in = nullptr;
				m_zdata.avail_in = 0;
				m_zdata.next_out = nullptr;
				m_zdata.avail_out = 0;
				int window_bits = ((uint)flags & (uint)Flags::SkipHeaders) ? -MAX_WBITS : MAX_WBITS;
				int sc = inflateInit2 (&m_zdata, window_bits);
				if (sc != Z_OK) DoThrow(sc);
			}

			inline DeflateStream::~DeflateStream()
			{
				inflateEnd (&m_zdata);
			}

			inline void DeflateStream::DoThrow(int ZLibStatusCode)
			{
				switch (ZLibStatusCode)
				{				
				case Z_STREAM_END: throw Exception("ZLib interface error."); break;			// All data decompressed successfully.  Shouldn't throw.
				case Z_OK: throw FormatException(S("Expected complete compressed file data within block."));
				case Z_NEED_DICT: throw NotSupportedException(S("Unable to decompress file: Dictionary needed."));
				case Z_DATA_ERROR: throw FormatException(S("Compressed data was corrupt."));
				case Z_STREAM_ERROR: throw NotSupportedException(S("Invalid interface to decompressor."));
				case Z_MEM_ERROR: throw OutOfMemoryException(S("Unable to decompress file: Out of memory."));
				default:
				case Z_BUF_ERROR: throw FormatException();
				}
			}

			inline void DeflateStream::GenerateOutput()
			{
				if (m_EOO) return;
				for (;;)
				{
					if (m_zdata.avail_in == 0 && !m_EOI)
					{
						m_InputBuffer.EnsureCapacity(BlockSize);
						m_InputBuffer.SetLength(m_pUnderlying->Read(m_InputBuffer.GetDirectAccess(0), m_InputBuffer.GetCapacity()));						
						if (m_InputBuffer.GetLength() == 0) m_EOI = true;
						m_InputBuffer.Rewind();
					}

					uint StartAvailInput = (uint)MinOf((Int64)UInt32_MaxValue, m_InputBuffer.GetLength() - m_InputBuffer.GetPosition());
					uint StartAvailOutput = (uint)MinOf((Int64)UInt32_MaxValue, m_OutputBuffer.GetCapacity());
					m_zdata.next_in = m_InputBuffer.GetDirectAccess();
					m_zdata.avail_in = StartAvailInput;
					m_zdata.next_out = m_OutputBuffer.GetDirectAccess(0);
					m_zdata.avail_out = StartAvailOutput;

					int sc = inflate(&m_zdata, Z_SYNC_FLUSH);
					switch (sc)
					{
					case Z_OK: break;
					case Z_STREAM_END: m_EOO = true; break;
					case Z_BUF_ERROR: 
						if (m_zdata.avail_in == 0) 
						{
							if (m_EOI) throw EndOfStreamException("Input data terminated before decompression was completed.");
							continue;
						}
						if (m_zdata.avail_out < StartAvailOutput) break;		// We made progress, let's continue.  zlib needed more output space.
						// Fall-through to an error...
					default: DoThrow(sc);
					}
					
					m_InputBuffer.Seek(StartAvailInput - m_zdata.avail_in, SeekOrigin::Current);
					m_OutputBuffer.SetLength(StartAvailOutput - m_zdata.avail_out);
					m_OutputBuffer.Rewind();
					if (m_OutputBuffer.GetLength() == 0) throw Exception("Unable to process additional decompression in stream.");
					return;
				}
			}

			inline int DeflateStream::ReadByte() override
			{				
				int ret = m_OutputBuffer.ReadByte();
				if (ret == -1) { GenerateOutput(); ret = m_OutputBuffer.ReadByte(); }
				return ret;
			}

			inline Int64 DeflateStream::Read(void *pBuffer, Int64 nLength) override
			{
				Int64 Bytes = m_OutputBuffer.Read(pBuffer, nLength);
				if (Bytes < nLength)
				{
					GenerateOutput();
					nLength -= Bytes;
					Int64 MoreBytes = m_OutputBuffer.Read((byte*)pBuffer + Bytes, nLength);
					return Bytes + MoreBytes;
				}
				return Bytes;
			}
		}
	}
}

#elif defined(DeflateStream_InternImpl)
	#include "DeflateStream_InternalImpl.h"
#endif

#endif	// __WBDeflateStream_h__

//	End of DeflateStream.h

