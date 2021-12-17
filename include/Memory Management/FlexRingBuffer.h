/////////
//	FlexRingBuffer.h
//	Copyright (C) 2001, 2014 by Wiley Black
/////////
//	Possible Improvements:
//
//	- The FlexRingBuffer class is not generally multithreaded-safe, and careful
//	  consideration may enable a version of the class that is thread-safe more
//	  efficiently than blanket object access locking.
//	  
/////////

#ifndef __FlexRingBuffer_h__
#define __FlexRingBuffer_h__

/**	Dependencies **/

#include "wbFoundation.h"

#include <assert.h>

#if defined(__GNUC__)
#include <malloc.h>
#elif defined(DOS) || defined(Win16)
#include <Alloc.h>
#endif

namespace wb
{
	namespace memory
	{
		/** FlexRingBuffer **/

		class FlexRingBuffer 
		{
		protected:
			byte*		m_pBuffer;
			uint		m_nAlloc;

				// Buffer is empty if these values are equal.
			uint		m_iHead;			// Data is read from the head, and then removed from the ring.
			uint		m_iTail;			// Data is added at the tail, and the length increases.

			bool MakeSpace(uint nBytes);

		protected:

			virtual uint GetInitialAllocate() const { return 1024; }
			virtual uint GetDeltaAllocate(uint nMinimum) const { return 1024 + nMinimum; }
	
				// OnAlloc() performs the actual re-allocation of memory.
				// If overriden in a flexible-sized ring buffer class, the base class member
				// should be called.  
			virtual bool OnAllocate(uint dwNewAlloc);

				// OnBufferFull() is called if the buffer is full.  In a fixed-size buffer, this
				// function can be used to empty some of the buffer, and a last re-attempt will be
				// made to continue without re-allocating or flushing.  
			virtual void OnBufferFull(){ }

				// OnPreBufferFlush() is called immediately before the buffer is going to be flushed.
				// Empty() is called to actually flush the buffer.
				// OnBufferFlushed() is called immediately after the buffer is flushed.		
			virtual void OnPreBufferFlush(){ }

				// OnBufferFlushed() is called if the buffer has overflowed, allocation is not possible,
				// and the contents of the buffer have been flushed.
			virtual void OnBufferFlushed(){ }

		public:
			FlexRingBuffer();
			~FlexRingBuffer();		

			virtual void Free();		// Frees all memory in the buffer
			virtual void Empty();		// Does not release memory, but marks buffer as empty
			virtual bool IsEmpty() const;
			virtual bool IsFull() const;

			void Add(byte b);			
			void Add(void *pBuffer, uint dwLenBytes);			

			/// <summary>Retrieves the number of bytes currently stored.  That is, bytes which can be retrieved by 
			/// calls to Get().</summary>
			virtual uint GetLength() const;

			virtual uint GetAlloc() const;							// # of bytes currently allocated
			virtual uint GetAvailable() const;						// # of bytes available for additional storage without memory allocation.
			virtual uint GetMaximumAvailable() const;				// # of bytes available for additional storage with allowable memory allocations.
			virtual uint GetMaximumAllocate() const;				// Override to limit memory usage...

			void Get(byte& b);
			void Get(void *pBuffer, uint dwLenBytes);
			void GetSkip(uint dwSkipBytes);
		};

		/////////
		//	Inline Functions
		//

		inline FlexRingBuffer::FlexRingBuffer()
		{
				// The ring buffer memory is allocated upon first use.
				// See MakeSpace().
			m_pBuffer	= NULL;
			m_nAlloc	= 0;
			m_iHead		= m_iTail	= 0;
		}

		inline FlexRingBuffer::~FlexRingBuffer()
		{
			Free();
		}

		inline void FlexRingBuffer::Free()
		{
			Empty();

			#if defined(_WINDOWS)
			if( m_pBuffer ) HeapFree( GetProcessHeap(), 0, m_pBuffer );
			#elif defined(__GNUC__)
			if( m_pBuffer ) free( m_pBuffer );	
			#else
			#	error No platform support here.
			#endif

			m_pBuffer = NULL;
			m_nAlloc = 0;
		}

		inline uint FlexRingBuffer::GetMaximumAllocate() const
		{
			return (UINT_MAX - 10UL);
		}

		inline void FlexRingBuffer::Empty(){ m_iHead = m_iTail = 0; }
		inline bool FlexRingBuffer::IsEmpty() const { return m_iHead == m_iTail; }
		inline bool FlexRingBuffer::IsFull() const { return GetLength() + 1 >= GetAlloc(); }

		inline uint FlexRingBuffer::GetAlloc() const { return m_nAlloc; }

		inline uint FlexRingBuffer::GetMaximumAvailable() const { 
			if (!GetMaximumAllocate()) return 0;
				// The +1 and -1 are to accomodate the ring buffer's requirement that the
				// head == tail case uses up one byte otherwise available.
			assert( GetMaximumAllocate() >= GetLength() + 1 );		// Somebody's lying, or we are hurting memory.
			return GetMaximumAllocate() - GetLength() - 1; 
		}

		inline uint FlexRingBuffer::GetAvailable() const { 
			if (!GetAlloc()) return 0;
				// The +1 and -1 are to accomodate the ring buffer's requirement that the
				// head == tail case uses up one byte otherwise available.
			assert( GetAlloc() >= GetLength() + 1 );				// Somebody's lying, or we are hurting memory.
			return GetAlloc() - GetLength() - 1; 
		}

		inline uint FlexRingBuffer::GetLength() const
		{
			if (m_iTail < m_iHead) return (GetAlloc() - m_iHead) + m_iTail;		// Wraps-around
			else return m_iTail - m_iHead;											// Cache-style or empty
		}

		inline void FlexRingBuffer::Add( byte b )
		{
			if( MakeSpace( 1 ) )
			{
				assert( m_pBuffer );
				assert( m_iTail < m_nAlloc );
				*(byte*)(m_pBuffer + m_iTail) = b;
				m_iTail ++;
				if (m_iTail >= GetAlloc()) m_iTail = 0;
			}
		}		

		inline void FlexRingBuffer::Get( byte& b )
		{
			assert( m_pBuffer );
			assert( m_iHead != m_iTail );		// Attempt to read data when GetLength() == 0!
			if( m_iHead == m_iTail ) return;	// No data, see assertion above.
	
			assert( m_iHead < m_nAlloc );
			b = *(byte*)(m_pBuffer + m_iHead);
			m_iHead ++;
			if( m_iHead >= GetAlloc() ) m_iHead = 0;
		}				

		inline bool FlexRingBuffer::MakeSpace(uint nBytes)
		{
			assert(nBytes > 0);

				// We check greater than or EQUAL TO, because the ring buffer cannot have head == tail (1 byte lost).
			if (GetLength() + nBytes >= GetAlloc()){

				OnBufferFull();
				if (GetLength() + nBytes >= GetAlloc()){

					// We need to allocate more memory to fit this new byte...

					uint nGrowBy;
					if( m_pBuffer ) nGrowBy = GetDeltaAllocate(nBytes);
					else nGrowBy = GetInitialAllocate();

					uint dwNewAlloc = GetAlloc() + nGrowBy;
					uint dwMaximumAllocate = GetMaximumAllocate();

						// First, check if this is more than we are allowed to allocate.  If it is,
						// will allocating exactly the maximum be sufficient?
					if (dwNewAlloc > dwMaximumAllocate && dwMaximumAllocate > GetLength() + nBytes)
						dwNewAlloc = dwMaximumAllocate;

						// The second condition prevents a wrap-around and logic errors...
						// The first condition verifies that we don't exceed our maximum.
					if (dwNewAlloc > dwMaximumAllocate || dwNewAlloc <= GetAlloc())
					{
						OnPreBufferFlush();
						Empty();
						OnBufferFlushed();
						assert(!(GetLength() + nBytes < GetAlloc()) || m_pBuffer != NULL);
						return (GetLength() + nBytes < GetAlloc());
					}

						// By logic, at this point dwMaximumAllocate MUST be greater than
						// or equal to dwNewAlloc.
						// We check if we are very near the maximum, and if near we just use that instead.
					if (dwMaximumAllocate - dwNewAlloc < 256) dwNewAlloc = dwMaximumAllocate;

					// We've received permission to try to allocate more memory...

					if (!OnAllocate( dwNewAlloc )) return false;
				}

					// Keep in mind that the ring buffer cannot have head == tail, thus we lose 1 byte.
				assert(GetLength() + nBytes < GetAlloc());
			}

			assert( m_pBuffer );
			return true;
		}

		inline bool FlexRingBuffer::OnAllocate(uint dwNewAlloc)
		{
			if (m_iTail < m_iHead){											// Wraps-around

					// The buffer wraps-around, so it won't work to just allocate more memory (this
					// would allocate more memory in the middle of the current buffer, creating
					// uninitalized data in a bad place.  So, we have to copy the entire buffer.
				#if defined(_WINDOWS)
				byte *pNewBuffer = (byte *)HeapAlloc( GetProcessHeap(), 0, dwNewAlloc );
				#elif defined(__GNUC__)
				byte *pNewBuffer = (byte *)malloc( dwNewAlloc );		
				#else
				#	error No platform support here.
				#endif

				assert(pNewBuffer);
				if (!pNewBuffer) return false;

				if (m_pBuffer){
					assert(GetAlloc() <= dwNewAlloc);
					CopyMemory(pNewBuffer, (m_pBuffer + m_iHead), GetAlloc() - m_iHead);

					assert(m_iTail < GetAlloc());
					assert(GetAlloc() - m_iHead + m_iTail <= dwNewAlloc);
					CopyMemory(pNewBuffer + (GetAlloc() - m_iHead), m_pBuffer, m_iTail);

					#if defined(_WINDOWS)
					HeapFree( GetProcessHeap(), 0, m_pBuffer );
					#elif defined(__GNUC__)
					free( m_pBuffer );
					#else
					#	error No platform support here.
					#endif
				}

				m_pBuffer = pNewBuffer;
				m_iTail += (GetAlloc() - m_iHead);
				m_iHead = 0;
				m_nAlloc = dwNewAlloc;

				assert(m_iTail < m_nAlloc);
				return true;
			}
			else {		// Cache-style or empty
					// The buffer is cache-style, so the end of the buffer is not included in
					// the current data set - meaning we can use a realloc() call instead of
					// copying the data.

				#if defined(_WINDOWS)
				if (!m_pBuffer) m_pBuffer = (byte *)HeapAlloc( GetProcessHeap(), 0, dwNewAlloc );
				else m_pBuffer = (byte *)HeapReAlloc( GetProcessHeap(), 0, m_pBuffer, dwNewAlloc );
				#elif defined(__GNUC__)
				if (!m_pBuffer) m_pBuffer = (byte *)malloc( dwNewAlloc );
				else m_pBuffer = (byte *)realloc( m_pBuffer, dwNewAlloc );
				#else
				#	error No platform support here.
				#endif

				assert(m_pBuffer);
				if (!m_pBuffer){
					m_nAlloc = m_iHead = m_iTail = 0;
					return false;
				}

				m_nAlloc = dwNewAlloc;
				return true;
			}
		}

		inline void FlexRingBuffer::Add(void *pBuffer, uint dwLenBytes)
		{
			if( MakeSpace( dwLenBytes ) ){
				if( m_iTail < m_iHead ){		// Wrap-around

					assert( m_iTail + dwLenBytes <= m_nAlloc );
					CopyMemory( m_pBuffer + m_iTail, pBuffer, dwLenBytes );

					m_iTail += dwLenBytes;
					if( m_iTail >= GetAlloc() ) m_iTail = 0;
				}
				else {							// Cache-style

					uint dwFirstLen = MinOf( dwLenBytes, GetAlloc() - m_iTail );

					assert( m_iTail + dwFirstLen <= m_nAlloc );
					CopyMemory( m_pBuffer + m_iTail, pBuffer, dwFirstLen );

					m_iTail += dwFirstLen;
					if( m_iTail >= GetAlloc() ) m_iTail = 0;

					if( dwFirstLen < dwLenBytes ){

							// At this point, we must now be wrap-around style.  This is because
							// the (m_iTail >= GetAlloc()) statement MUST have been true, since 
							// for dwFirstLen to be anything other than exactly equal to dwLenBytes, 
							// the MIN would have had to have chosen GetAlloc() - m_iTail, and so:
							//		m_iTail += dwFirstLen;
							//		m_iTail += (GetAlloc() - m_iTail);
							//		m_iTail = m_iTail + GetAlloc() - m_iTail;
							//		m_iTail = GetAlloc();
							//		Then, m_iTail = 0;
							// Finally, m_iHead cannot be equal to zero because MakeSpace() has
							// assured us that this data will fit properly.

						assert( m_iTail < m_iHead );		// Verify we are wrap-around style & not empty.
						assert( m_iTail + (dwLenBytes - dwFirstLen) <= m_nAlloc );

						CopyMemory( m_pBuffer + m_iTail, (byte *)pBuffer + dwFirstLen, dwLenBytes - dwFirstLen );
						m_iTail += (dwLenBytes - dwFirstLen);
						if( m_iTail >= GetAlloc() ) m_iTail = 0;
					}
				}
			}
		}

		inline void FlexRingBuffer::Get(void *pBuffer, uint dwLenBytes)
		{ 
			assert(m_pBuffer);
			assert(GetLength() >= dwLenBytes);		// Attempt to read data when GetLength() < dwLenBytes!

			if (GetLength() >= dwLenBytes){
				if (m_iTail < m_iHead){		// Wrap-around

					uint dwFirstLen = MinOf(dwLenBytes, GetAlloc() - m_iHead);

					assert(m_iHead + dwFirstLen <= m_nAlloc);
					CopyMemory(pBuffer, m_pBuffer + m_iHead, dwFirstLen);

					m_iHead += dwFirstLen;
					if (m_iHead >= GetAlloc()) m_iHead = 0;

					if (dwFirstLen < dwLenBytes){

							// At this point, we must now be cache-style.  This is because
							// the (m_iHead >= GetAlloc()) statement MUST have been true, since 
							// for dwFirstLen to be anything other than exactly equal to dwLenBytes, 
							// the MIN would have had to have chosen GetAlloc() - m_iHead, and so:
							//		m_iHead += dwFirstLen;
							//		m_iHead += (GetAlloc() - m_iHead);
							//		m_iHead = m_iHead + GetAlloc() - m_iHead;
							//		m_iHead = GetAlloc();
							//		Then, m_iHead = 0;
							// Finally, m_iTail cannot be equal to zero because MakeSpace() has
							// assured us that this data is available.

						assert(m_iHead < m_iTail);		// Verify we are cache-style & not empty.
						assert(m_iHead + (dwLenBytes - dwFirstLen) <= m_nAlloc);

						CopyMemory((byte *)pBuffer + dwFirstLen, m_pBuffer + m_iHead, dwLenBytes - dwFirstLen);
						m_iHead += (dwLenBytes - dwFirstLen);
						if (m_iHead >= GetAlloc()) m_iHead = 0;
					}
				}
				else {							// Cache-style
					assert(m_iHead + dwLenBytes <= m_nAlloc);
					CopyMemory(pBuffer, m_pBuffer + m_iHead, dwLenBytes);
					m_iHead += dwLenBytes;
					if (m_iHead >= GetAlloc()) m_iHead = 0;
				}
			}
		}		

		inline void FlexRingBuffer::GetSkip(uint dwLenBytes)
		{ 
			assert(m_pBuffer);
			assert(GetLength() >= dwLenBytes);		// Attempt to skip data when GetLength() < dwLenBytes!

			if (GetLength() >= dwLenBytes){
				if (m_iTail < m_iHead){		// Wrap-around

					uint dwFirstLen = MinOf(dwLenBytes, GetAlloc() - m_iHead);

					assert(m_iHead + dwFirstLen <= m_nAlloc);					

					m_iHead += dwFirstLen;
					if (m_iHead >= GetAlloc()) m_iHead = 0;

					if (dwFirstLen < dwLenBytes){

							// At this point, we must now be cache-style.  This is because
							// the (m_iHead >= GetAlloc()) statement MUST have been true, since 
							// for dwFirstLen to be anything other than exactly equal to dwLenBytes, 
							// the MIN would have had to have chosen GetAlloc() - m_iHead, and so:
							//		m_iHead += dwFirstLen;
							//		m_iHead += (GetAlloc() - m_iHead);
							//		m_iHead = m_iHead + GetAlloc() - m_iHead;
							//		m_iHead = GetAlloc();
							//		Then, m_iHead = 0;
							// Finally, m_iTail cannot be equal to zero because MakeSpace() has
							// assured us that this data is available.

						assert(m_iHead < m_iTail);		// Verify we are cache-style & not empty.
						assert(m_iHead + (dwLenBytes - dwFirstLen) <= m_nAlloc);
						
						m_iHead += (dwLenBytes - dwFirstLen);
						if (m_iHead >= GetAlloc()) m_iHead = 0;
					}
				}
				else {							// Cache-style
					assert(m_iHead + dwLenBytes <= m_nAlloc);
					m_iHead += dwLenBytes;
					if (m_iHead >= GetAlloc()) m_iHead = 0;
				}
			}
		}
	}
}

#endif	// __FlexRingBuffer_h__

//	End of FlexRingBuffer.h



