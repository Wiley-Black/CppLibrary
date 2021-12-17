/////////
//	FlexRingBufferEx.h
//	Copyright (C) 2001, 2014 by Wiley Black
/////////
//	Based on FlexRingBuffer, the FlexRingBufferEx class
//	adds the capability of pseudo-random access (for purposes of
//	adding data only) to the ring buffer.
/////////

#ifndef __FlexRingBufferEx_h__
#define __FlexRingBufferEx_h__

/////////
//	Dependencies
//

#include "Memory Management/FlexRingBuffer.h"

namespace wb
{
	namespace memory
	{
		/** FlexRingBufferEx **/

		class FlexRingBufferEx : public FlexRingBuffer
		{
				// AdjustHeadIndices() takes on two purposes.  First, it ensures the
				// requested number of bytes exist in the head---tail window, and if
				// not it returns false.  Secondly, it translates iIndex from a 
				// zero-based index into an index appropriate for the ring buffer.
			bool AdjustHeadIndices(uint& iIndex, uint nBytes);

				// AdjustTailIndices() takes on two purposes.  First, it ensures the
				// requested number of bytes will fit, just like MakeSpace(),
				// but based on iIndex.  Secondly, if the index for the add-at
				// operation falls outside or partially outside the head---tail
				// window, then AdjustTailIndices() moves the tail to include this 
				// new data.  If either of these goals fail, AdjustIndices()
				// returns false.
			bool AdjustTailIndices(uint& iIndex, uint nBytes);
		public:

				/** Pseudo-Random-Access Ring Buffer **/

			void AddAt(uint iIndex, byte b );			
			void AddAt(uint iIndex, void *pBuffer, uint dwLenBytes);

			void PeekAt(uint iIndex, byte& b );			
			void PeekAt(uint iIndex, void *pBuffer, uint dwLenBytes);

				/** Direct buffer access **/

				/** DirectAdd() and AddedDirect()
					Use these methods to add data directly to a ring buffer.  These methods do not 
					enlarge (flex) the buffer.  DirectAdd() is called to retrieve a pointer and space
					available count for direct add access to the buffer.  AddedDirect() is called after
					the data has been added (by caller) to let the ring buffer know how many bytes were
					added.

					dwSpace:	This value is returned.  It indicates how much space may be added at the
								returned pointer address.

					Direct access to the ring buffer is a multiple-step process.

					1. DirectAdd().  Add the first 'dwSpace' (or less) of data using the returned pointer.
					2. AddedDirect(n), where 'n' is number of bytes you have added at the pointer.
					3. If n was sufficient for all data needing to be added, done.
					4. DirectAdd().  Add the next 'dwSpace' of data using the returned pointer.
					5. AddedDirect(n), where 'n' is number of bytes you have added at the pointer.

					If no Get() calls occur and no allocation methods are used, then additional repetitions 
					of the process will not return additional 'dwSpace'.  It is possible that both or the 
					second DirectAdd() call will have no space available.  To predict the total space 
					available, use GetFree().  If the first DirectAdd() call returns zero space, a following
					call will also return zero space and can be skipped (assuming no other methods intermixed).

					The space available in the second DirectAdd() call will not have increased from the
					first call if AddedDirect() did not consume all available space in the first call.  If
					such a scenario occurs, more than two repetitions of the process may be needed, and an
					'until space is zero' algorithm is suggested instead of the fixed two-repition method.

					The DirectAdd() function can be called and AddedDirect() not called, in the event
					that the pointer and/or space-count is needed before it is known if data will actually
					be added.

					AddedDirect() calls of zero bytes are allowed and inexpensive.
				**/
			void *DirectAdd(uint& dwSpace) const;
			void AddedDirect(uint dwBytes);
		};

		/////////
		//	Inline Functions
		//

		inline void FlexRingBufferEx::AddAt(uint iIndex, byte b)
		{
			if (!AdjustTailIndices( iIndex, 1 )) return;

			assert(m_pBuffer);
			assert(iIndex < m_nAlloc);
			*(byte*)(m_pBuffer + iIndex) = b;
		}

		/////////

		inline bool FlexRingBufferEx::AdjustHeadIndices(uint& iIndex, uint nBytes)
		{
			if (iIndex + nBytes > GetLength()) return false;

				// Convert iIndex from a zero-based index into an index into our
				// ring buffer.  We have already verified (above) that the resulting
				// value will be valid and fit within the head---tail window.

			iIndex += m_iHead;
			if( iIndex >= GetAlloc() ) iIndex -= GetAlloc();
			assert( iIndex < GetAlloc() );

			return true;
		}

		inline void FlexRingBufferEx::PeekAt(uint iIndex, byte& b)
		{
			if (!AdjustHeadIndices( iIndex, 1 )) return;

			assert(m_pBuffer);
			assert(iIndex < m_nAlloc);
			b = *(byte*)(m_pBuffer + iIndex);
			return;
		}		

			/** Direct-Access **/

		inline void *FlexRingBufferEx::DirectAdd(uint& dwSpace ) const
		{
			if( m_iTail < m_iHead ){		// Wrap-around

				dwSpace = m_iHead - m_iTail - 1;

			} 
			else {							// Cache-Style

				if( m_iHead ) dwSpace = m_nAlloc - m_iTail;
				else dwSpace = m_nAlloc - m_iTail - 1;
			}

			assert( m_pBuffer );
			assert( dwSpace <= GetAvailable() );
			assert( m_iTail < m_nAlloc );
			return m_pBuffer + m_iTail;
		}

		inline void FlexRingBufferEx::AddedDirect(uint dwBytes)
		{
			assert( GetAvailable() >= dwBytes );

			assert( m_pBuffer );
			assert( m_iTail < m_nAlloc );
			assert( m_iTail + dwBytes <= m_nAlloc );			// Assertion: Caller has written to memory past end-of-buffer!
			m_iTail += dwBytes;
			if( m_iTail >= GetAlloc() ) m_iTail = 0;
			assert( dwBytes == 0 || m_iTail != m_iHead );
		}

			/** Core implementations **/

		inline bool FlexRingBufferEx::AdjustTailIndices(uint& iIndex, uint nBytes)
		{
			/** iIndex is a zero-based index.  It is not an index into the ring buffer
				yet, because it is based on zero being the start of its sequence.  One
				of the goals of this function is to adjust iIndex to be an index into
				the ring buffer, i.e. if the head is at 10, and the iIndex is 0 at the
				start of this function, at return iIndex should be 10. **/

			if (iIndex + nBytes > GetLength())
			{	
					// We are adding data at a point outside the head---tail window,
					// so we have to move the tail out to include this region.

				if( !MakeSpace( iIndex + nBytes - GetLength() ) ) return false;
				m_iTail += iIndex + nBytes - GetLength();
				if( m_iTail >= GetAlloc() ) m_iTail -= GetAlloc();
				assert( m_iTail < GetAlloc() );
			}

				// Convert iIndex from a zero-based index into an index into our
				// ring buffer.  We have already verified (above) that the resulting
				// value will be valid and fit within the head---tail window.

			iIndex += m_iHead;
			if( iIndex >= GetAlloc() ) iIndex -= GetAlloc();
			assert( iIndex < GetAlloc() );

			return true;
		}

		inline void FlexRingBufferEx::AddAt(uint iIndex, void* pBuffer, uint dwLenBytes)
		{
			if( AdjustTailIndices( iIndex, dwLenBytes ) ){
				if( iIndex < m_iHead ){			// Wrap-around
					assert( iIndex + dwLenBytes <= m_nAlloc );
					CopyMemory( m_pBuffer + iIndex, pBuffer, dwLenBytes );
					iIndex += dwLenBytes;
					if( iIndex >= GetAlloc() ) iIndex = 0;
				}
				else {							// Cache-style
					uint dwFirstLen = MinOf( dwLenBytes, GetAlloc() - iIndex );

					assert( iIndex + dwFirstLen <= m_nAlloc );
					CopyMemory( m_pBuffer + iIndex, pBuffer, dwFirstLen );

					if( dwFirstLen < dwLenBytes ){

						iIndex += dwFirstLen;
						if( iIndex >= GetAlloc() ) iIndex = 0;

							// At this point, we must now be wrap-around style.  This is because
							// the (iIndex >= GetAlloc()) statement MUST have been true, since 
							// for dwFirstLen to be anything other than exactly equal to dwLenBytes, 
							// the MIN would have had to have chosen GetAlloc() - iIndex, and so:
							//		iIndex += dwFirstLen;
							//		iIndex += (GetAlloc() - iIndex);
							//		iIndex = iIndex + GetAlloc() - iIndex;
							//		iIndex = GetAlloc();
							//		Then, iIndex = 0;
							// Finally, m_iHead cannot be equal to zero because AdjustTailIndices() 
							// has assured us that this data will fit properly.

						assert( iIndex < m_iHead );		// Verify we are wrap-around style & not empty.
						assert( iIndex + (dwLenBytes - dwFirstLen) <= m_nAlloc );

						CopyMemory( m_pBuffer + iIndex, (byte *)pBuffer + dwFirstLen, dwLenBytes - dwFirstLen );
						iIndex += (dwLenBytes - dwFirstLen);
						if( iIndex >= GetAlloc() ) iIndex = 0;
					}
				}
			}
		}

		inline void FlexRingBufferEx::PeekAt(uint iIndex, void* pBuffer, uint dwLenBytes)
		{
			if( !AdjustHeadIndices( iIndex, dwLenBytes ) ) return;

			assert( m_pBuffer );

			if( m_iTail < iIndex ){			// Wrap-around

				uint dwFirstLen = MinOf( dwLenBytes, GetAlloc() - iIndex );

				assert( iIndex + dwFirstLen <= m_nAlloc );
				CopyMemory( pBuffer, m_pBuffer + iIndex, dwFirstLen );

				if( dwFirstLen < dwLenBytes ){

					iIndex += dwFirstLen;
					if( iIndex >= GetAlloc() ) iIndex = 0;

						// At this point, we must now be cache-style.  This is because
						// the (iIndex >= GetAlloc()) statement MUST have been true, since 
						// for dwFirstLen to be anything other than exactly equal to dwLenBytes, 
						// the MIN would have had to have chosen GetAlloc() - iIndex, and so:
						//		iIndex += dwFirstLen;
						//		iIndex += (GetAlloc() - iIndex);
						//		iIndex = iIndex + GetAlloc() - iIndex;
						//		iIndex = GetAlloc();
						//		Then, iIndex = 0;
						// Finally, m_iTail cannot be equal to zero because AdjustHeadIndices() 
						// has assured us that this data is available.

					assert( iIndex < m_iTail );		// Verify we are cache-style & not empty.
					assert( iIndex + (dwLenBytes - dwFirstLen) <= m_nAlloc );

					CopyMemory( (byte *)pBuffer + dwFirstLen, m_pBuffer + iIndex, dwLenBytes - dwFirstLen );
				}
			}
			else {							// Cache-Style

					// The tail follows the head, so there is no "wrap-around" currently
					// present in the buffer.  And, AdjustHeadIndices() has assured
					// us that 'dwLenBytes' are available starting from 'iIndex', so
					// all we have to do is grab it in a nice linear fasion.

				assert( iIndex + dwLenBytes <= m_nAlloc );
				CopyMemory( pBuffer, m_pBuffer + iIndex, dwLenBytes );
			}
		}
	}
}

#endif	// __FlexRingBufferEx_h__

//	End of FlexRingBufferEx.h

