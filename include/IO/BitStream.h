/////////
//	BitStream.h
//	Copyright (C) 2014 by Wiley Black
//
//	Provides bitwise access to a stream.
////

#ifndef __WBBitStream_h__
#define __WBBitStream_h__

/** Table of contents **/

namespace wb {
	namespace io {
		class LSBBitStream;
	}
}

/** Dependencies **/

#include "../wbFoundation.h"
#include "../Foundation/STL/Memory.h"
#include "Streams.h"
#include "../Memory Management/Allocation.h"

/** Content **/

namespace wb
{
	namespace io
	{		
		using namespace ::wb::memory;

		/// <summary>
		/// The LSBBitStream class supports reading only in LSB-first-order.  That is, the first 
		/// bit read is taken from the least-significant of the 8-bits of the first byte.  This
		/// is the directionality used by DEFLATE, but may not always be the case.  Note that
		/// this is not the same as endianness, which specifies the sequencing of bytes within
		/// a multibyte value.
		/// </summary>
		class LSBBitStream
		{
			/// <summary>New bits are added to the high bits and bits are read from the low bits.</summary>
			uint			m_BitBuffer;
			uint			m_BitCount;

		public:
			r_ptr<Stream>	m_pUnderlying;

			LSBBitStream(r_ptr<Stream>&& pStream) : m_pUnderlying(std::move(pStream)) { m_BitBuffer = 0; m_BitCount = 0; }			

			uint	ReadBits(uint Count)
			{
				while (m_BitCount < Count) 
				{
					int NewByte = m_pUnderlying->ReadByte();
					if (NewByte < 0) throw EndOfStreamException();
					m_BitBuffer |= (((uint)NewByte) << m_BitCount);
					m_BitCount += 8;
				}
				uint ret = m_BitBuffer & ((1U << Count) - 1U);
				m_BitBuffer >>= Count;
				m_BitCount -= Count;
				return ret;
			}

			void FlushCurrentByte() { m_BitCount = 0; }

			/// <summary>ReadByte() allows byte access to the stream, but will throw an exception if the
			/// read position is not presently byte-aligned.  FlushCurrentByte() can be used to move to the
			/// next byte-aligned boundary by discarding any remaining bits from the current byte.</summary>
			byte	ReadByte()
			{
				if (m_BitCount == 8)
				{
					m_BitCount = 0;
					return (byte)m_BitBuffer;
				}

				if (m_BitCount == 0)
				{
					int ret = m_pUnderlying->ReadByte();
					if (ret < 0) throw EndOfStreamException();
					return (byte)ret;
				}

				throw Exception("ReadByte() can only be used on byte-aligned boundaries.");
			}
		};
	}
}

#endif	// __WBBitStream_h__

//	End of BitStream.h

