/*	MemoryStream.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBMemoryStream_h__
#define __WBMemoryStream_h__

#include "Streams.h"
#include "../Memory Management/Buffer.h"

namespace wb
{
	namespace io
	{
		class MemoryStream : public Stream
		{
			wb::memory::Buffer	m_Buffer;
			UInt64	m_iPosition;
			UInt64	m_nLength;

			UInt64	GetNextCapacity(UInt64 Additional)
			{
				UInt64 ret = m_Buffer.GetSize();	// Start with current size.
				ret += Additional;						// Add bytes that we know we will need.
				if (ret < 256) return 256;
				if (ret < 1024) return 1024;
				if (ret < 4096) return 4096;
				if (ret < 16384) return 16384;
				if (ret < 65536) return 65536;
				if (ret < 262144) return 262144;
				if (ret < 1048576) return 1048576;		// 1MB
				return (ret & ~0xFFFFF) + 0x100000;		// Round up to next MB.
			}

		public:			
			MemoryStream(int capacity = 4096) : m_Buffer(capacity)
			{
				m_iPosition = 0;
				m_nLength = 0;
			}

			MemoryStream(const MemoryStream& cp)
			{
				m_Buffer.Alloc(cp.m_nLength);
				if (cp.m_nLength > UInt32_MaxValue) throw NotImplementedException();
				MoveMemory(m_Buffer.At(), cp.m_Buffer.At(), (UInt32)cp.m_nLength);		// Transfer only necessary data.
				m_iPosition = cp.m_iPosition;
				m_nLength = cp.m_nLength;
			}

			MemoryStream(MemoryStream&& cp) : m_Buffer(std::move(cp.m_Buffer))
			{
				m_iPosition = cp.m_iPosition;
				m_nLength = cp.m_nLength;
			}

			MemoryStream& operator=(const MemoryStream& cp)
			{
				if (cp.m_nLength > UInt32_MaxValue) throw NotImplementedException();
				if (m_Buffer.GetSize() < cp.m_nLength) m_Buffer.Alloc(cp.m_nLength);
				MoveMemory(m_Buffer.At(), cp.m_Buffer.At(), (UInt32)cp.m_nLength);		// Transfer only necessary data.
				m_iPosition = cp.m_iPosition;
				m_nLength = cp.m_nLength;
				return *this;
			}

			MemoryStream& operator=(MemoryStream&& mv)
			{
				m_Buffer.Free();
				m_Buffer = std::move(mv.m_Buffer);
				m_iPosition = mv.m_iPosition;
				m_nLength = mv.m_nLength;
				return *this;
			}

			~MemoryStream() { Close(); }

			bool CanRead() override { return true; }
			bool CanWrite() override { return true; }
			bool CanSeek() override { return true; }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			int ReadByte() override 
			{
				if (m_iPosition >= m_nLength) return -1;
				int ret = ((byte*)m_Buffer.At())[m_iPosition];
				m_iPosition++;
				return ret;
			}
			Int64 Read(void* pBuffer, Int64 nLength) override 
			{
				Int64 Available = (Int64)(m_nLength - m_iPosition);
				if (nLength > Available) nLength = Available;
				if (nLength > UInt32_MaxValue) throw NotImplementedException();
				MoveMemory (pBuffer, ((byte*)m_Buffer.At()) + m_iPosition, (UInt32)nLength);
				m_iPosition += nLength;
				return nLength;
			}

			void WriteByte(byte ch) override 
			{ 
				if (m_iPosition >= m_Buffer.GetSize()) m_Buffer.Realloc(GetNextCapacity(1));
				((byte*)m_Buffer.At())[m_iPosition] = ch;
				m_iPosition ++;
				if (m_iPosition > m_nLength) m_nLength++;
			}			
			void Write(const void *pBuffer, Int64 nLength) override
			{
				if (m_iPosition + nLength > m_Buffer.GetSize()) m_Buffer.Realloc(GetNextCapacity(nLength));
				if (nLength > UInt32_MaxValue) throw NotImplementedException();
				MoveMemory (((byte*)m_Buffer.At()) + m_iPosition, pBuffer, (UInt32)nLength);
				m_iPosition += nLength;
				if (m_iPosition > m_nLength) m_nLength = m_iPosition;
			}

			Int64 GetPosition() const override { return m_iPosition; }
			Int64 GetCapacity() const { return m_Buffer.GetSize(); }
			Int64 GetLength() const override { return m_nLength; }

			void Seek(Int64 offset, SeekOrigin origin) override 
			{
				switch (origin)
				{
				case SeekOrigin::Begin: m_iPosition = offset; return;
				case SeekOrigin::Current: m_iPosition += offset; return;
				case SeekOrigin::End: m_iPosition = m_nLength + offset; return;
				default: throw ArgumentException(S("Invalid origin."));
				}
			}

			void Close()
			{
			}

			void Rewind() { m_iPosition = 0; }

			/** Retrieves a pointer to the current position of the buffer **/
			byte* GetDirectAccess() { return ((byte*)m_Buffer.At()) + m_iPosition; }
			byte* GetDirectAccess(UInt64 AtPosition) { return ((byte*)m_Buffer.At()) + AtPosition; }
			const byte* GetDirectAccess() const { return (const byte*)((byte*)m_Buffer.At()) + m_iPosition; }
			const byte* GetDirectAccess(UInt64 AtPosition) const { return (const byte*)((byte*)m_Buffer.At()) + AtPosition; }

			void EnsureCapacity(Int64 nBytes)
			{
				if (nBytes <= GetCapacity()) return;
				m_Buffer.Realloc(GetNextCapacity(nBytes - GetCapacity()));
			}

			void SetLength(Int64 nBytes) { EnsureCapacity(nBytes); m_nLength = nBytes; }
		};
	}
}

#endif	// __WBMemoryStream_h__

//	End of MemoryStream.h

