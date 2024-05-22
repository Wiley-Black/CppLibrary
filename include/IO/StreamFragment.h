/*	StreamFragment.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)

	Defines the StreamFragment class, which can be used as a wrapper to subsection a part of a larger stream.  For example,
	if there is an "archive" FileStream that contains multiple files within the larger archive, certain regions of the file
	can be assigned to represent certain files.  The StreamFragment class provides a mechanism to define the start and end/length
	positions within a larger stream and defines them as an individual stream.

	A StreamFragment class is never writable even if the underlying stream is.  The StreamFragment class is seekable if the
	underlying stream is seekable.  Flushing the StreamFragment will flush the underlying stream, but closing the StreamFragment
	has no effect.

	The StreamFragment class will generate an EOF with respect to the fragment subsection, preventing data from outside of the
	fragment from being accessed via the StreamFragment.  Data outside the fragment's boundaries can still be accessed through
	the underlying stream if the caller has access to that object, however care should be taken in controlling access between
	the objects - since the StreamFragment only represents a logical layer, it assumes that it is the only object accessing
	the underlying stream during its lifetime.  For a seekable stream, this can be overcome by performing a Seek() operation 
	after accessing the underlying stream.
*/

#ifndef __WBStreamFragment_h__
#define __WBStreamFragment_h__

#include "../wbFoundation.h"

namespace wb
{
	namespace io
	{
		class StreamFragment : public Stream
		{
			Stream*	m_pUnderlying;
			Int64	m_PositionOffset;		// If seekable, this provides the "start position" to offset all positions relative to the underlying stream.
			Int64	m_nLength;

			/// <summary>m_CurrentPosition tracks the current position within the logical fragment.  The current position within the underlying
			/// stream should be (m_PositionOffset + m_CurrentPosition) for a seekable stream (m_PositionOffset is not used for a non-seekable
			/// stream.)  The actual current position of the underlying stream may differ from this tracked current position if the underlying
			/// stream is accessed outside of the StreamFragment wrapper.  This results in an inconsistent state, but can be corrected with a
			/// Seek() call.
			Int64	m_CurrentPosition;

		public:
			/// <summary>This constructor can be used for a seekable stream only.</summary>
			StreamFragment(Stream& UnderlyingStream, Int64 StartPosition, Int64 Length);

			/// <summary>This constructor can be used for a non-seekable stream, or for a seekable-stream when the current position can be taken as
			/// the start of the stream fragment.</summary>
			StreamFragment(Stream& UnderlyingStream, Int64 Length);

			StreamFragment(const StreamFragment&);
			StreamFragment(StreamFragment&&);

			~StreamFragment() override { Close(); }

			bool CanRead() override { return m_pUnderlying->CanRead(); }
			bool CanWrite() override { return false; }
			bool CanSeek() override { return m_pUnderlying->CanSeek(); }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			int ReadByte() override;
			Int64 Read(void *pBuffer, Int64 nLength) override;

			void WriteByte(byte ch) override;
			void Write(const void *pBuffer, Int64 nLength) override;			

			Int64 GetPosition() const override { return m_CurrentPosition; }
			Int64 GetLength() const override { return m_nLength; }
			void Seek(Int64 offset, SeekOrigin origin) override;

			void Flush() override { m_pUnderlying->Flush(); }
			void Close() { }
		};		

		/** Implementation **/
		
		inline StreamFragment::StreamFragment(Stream& UnderlyingStream, Int64 StartPosition, Int64 Length)
		{
			m_pUnderlying = &UnderlyingStream;
			m_PositionOffset = StartPosition;
			m_nLength = Length;			

			if (!m_pUnderlying->CanSeek()) throw ArgumentException("This constructor can only be used with a seekable stream.");
			m_CurrentPosition = m_pUnderlying->GetPosition() - m_PositionOffset;
		}

		/// <summary>This constructor can be used for a non-seekable stream, or for a seekable-stream when the current position can be taken as
		/// the start of the stream fragment.</summary>
		inline StreamFragment::StreamFragment(Stream& UnderlyingStream, Int64 Length)
		{
			m_pUnderlying = &UnderlyingStream;
			m_nLength = Length;
			m_CurrentPosition = 0;

			if (m_pUnderlying->CanSeek()) m_PositionOffset = m_pUnderlying->GetPosition();
			else m_PositionOffset = 0;
		}

		inline StreamFragment::StreamFragment(const StreamFragment& cp)
		{
			m_pUnderlying = cp.m_pUnderlying;
			m_PositionOffset = cp.m_PositionOffset;
			m_nLength = cp.m_nLength;
			m_CurrentPosition = cp.m_CurrentPosition;
		}

		inline StreamFragment::StreamFragment(StreamFragment&& mv)
		{
			m_pUnderlying = mv.m_pUnderlying;
			m_PositionOffset = mv.m_PositionOffset;
			m_nLength = mv.m_nLength;
			m_CurrentPosition = mv.m_CurrentPosition;
		}

		inline int StreamFragment::ReadByte()
		{
			if (m_CurrentPosition >= 0 && m_CurrentPosition < m_nLength) { 
				int ret = m_pUnderlying->ReadByte();
				if (ret != -1) m_CurrentPosition ++;
				return ret;
			}
			return -1;
		}

		inline Int64 StreamFragment::Read(void *pBuffer, Int64 nLength)
		{
			if (m_CurrentPosition < 0 || m_CurrentPosition >= m_nLength) return 0;

			nLength = MinOf(m_nLength - m_CurrentPosition, nLength);
			Int64 Bytes = m_pUnderlying->Read(pBuffer, nLength);
			m_CurrentPosition += Bytes;
			return Bytes;
		}

		inline void StreamFragment::WriteByte(byte ch) { throw NotSupportedException(); }
		inline void StreamFragment::Write(const void *pBuffer, Int64 nLength) { throw NotSupportedException(); }

		inline void StreamFragment::Seek(Int64 offset, SeekOrigin origin)
		{
			switch (origin)
			{
			case SeekOrigin::Begin: m_pUnderlying->Seek(m_PositionOffset + offset, SeekOrigin::Begin); break;
			case SeekOrigin::Current: m_pUnderlying->Seek(offset, SeekOrigin::Current); break;
			case SeekOrigin::End: m_pUnderlying->Seek(m_PositionOffset + m_nLength + offset, SeekOrigin::Begin); break;
			}

			m_CurrentPosition = m_pUnderlying->GetPosition() - m_PositionOffset;
		}
	}
}

#endif	// __WBStreamFragment_h__

//	End of StreamFragment.h

