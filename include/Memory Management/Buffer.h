/////////
//	Buffer.h
//	Copyright (C) 2001, 2014 by Wiley Black
/////////

#ifndef __WBBuffer_h__
#define __WBBuffer_h__

/**	Dependencies **/

#include "../Platforms/Platforms.h"

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
		/** Buffer **/

		class Buffer 
		{
		protected:
			void*		m_pBuffer;
			UInt64		m_nSize;

		public:
			#ifdef _DEBUG
			UInt32 ValidCode;
			enum { TargetCode = 0x12345678 };
			#endif

			Buffer();
			Buffer(UInt64 nSize);
			Buffer(Buffer&&);
			Buffer(const Buffer&) { throw NotSupportedException(S("Use a move copy or Alloc() and direct copy instead.")); }
			~Buffer();
			
			Buffer& operator=(const Buffer&) { throw NotSupportedException(S("Use a move copy or Alloc() and direct copy instead.")); }
			Buffer& operator=(Buffer&& mv);

			/// <summary>Alloc() allocates a buffer of size nSize, if not already allocated.  Although the existing memory may or may not
			/// be used, no content is copied during the allocation.</summary>
			void Alloc(UInt64 nSize);

			/// <summary>Realloc() allocates a buffer of size nSize, retaining all existing data in the buffer.</summary>
			void Realloc(UInt64 nSize);

			/// <summary>Realloc() allocates a buffer of size nSize.  Some existing data from the buffer is retained, beginning at
			/// the index specified by Start (inclusive) and continuing for Length bytes.</summary>
			void Realloc(UInt64 nSize, UInt64 Start, UInt64 Length);

			void Free();

			UInt64 GetSize() const { return m_nSize; }
			#if 0
			operator void*() { return (void*)m_pBuffer; }
			operator const void*() const { return (const void*)m_pBuffer; }
			operator byte*() { return (byte*)m_pBuffer; }
			operator const byte*() const { return (const byte*)m_pBuffer; }
			#endif
			void* At() { return (void *)m_pBuffer; }
			const void* At() const { return (const void *)m_pBuffer; }
		};

		/////////
		//	Inline Functions
		//

		inline Buffer::Buffer()
		{
			m_pBuffer	= nullptr;
			m_nSize		= 0;
			#ifdef _DEBUG
			ValidCode = TargetCode;
			#endif
		}

		inline Buffer::Buffer(UInt64 nCapacity)
		{
			m_pBuffer	= nullptr;
			m_nSize		= 0;
			#ifdef _DEBUG
			ValidCode = TargetCode;
			#endif

			Alloc(nCapacity);
		}

		inline Buffer::Buffer(Buffer&& mv)
		{
			m_pBuffer	= mv.m_pBuffer;
			m_nSize		= mv.m_nSize;
			mv.m_pBuffer = nullptr;
			mv.m_nSize = 0;
			#ifdef _DEBUG
			ValidCode = TargetCode;
			#endif
		}

		inline Buffer& Buffer::operator=(Buffer&& mv)
		{
			#ifdef _DEBUG
			ValidCode = TargetCode;
			#endif
			if (m_pBuffer != nullptr) Free();
			m_pBuffer	= mv.m_pBuffer;
			m_nSize		= mv.m_nSize;
			mv.m_pBuffer	= nullptr;
			mv.m_nSize		= 0;
			return *this;
		}

		inline Buffer::~Buffer()
		{
			Free();
		}

		inline void Buffer::Free()
		{
			#ifdef _DEBUG
			assert (ValidCode == TargetCode);
			#endif

			#if 0
			if (m_pBuffer) { delete[] (byte*)m_pBuffer; }
			#elif 1
			if (m_pBuffer) free( m_pBuffer );	
			#else
				#if defined(_WINDOWS)
				if (m_pBuffer) HeapFree( GetProcessHeap(), 0, m_pBuffer );
				#elif defined(__GNUC__)
				if (m_pBuffer) free( m_pBuffer );	
				#else
				#	error No platform support here.
				#endif
			#endif

			m_pBuffer = nullptr;
			m_nSize = 0;
		}		

		inline void Buffer::Alloc(UInt64 nCapacity)
		{
			if (nCapacity > UInt32_MaxValue) throw NotImplementedException();			

			// Since Alloc() has no requirement to retain the existing data (see Realloc for that),
			// it will probably be faster to free and allocate a new block.  This prevents the OS
			// from performing a large, unnecessary copy operation should we call realloc() and
			// the OS not have the contiguous buffer space available.
			Free();

			if (nCapacity > 0)
			{
				#if 0
				m_pBuffer = new byte [(UInt32)nCapacity];
				#elif 1
				m_pBuffer = (byte *)malloc((UInt32) nCapacity);	
				#else
					#if defined(_WINDOWS)
					m_pBuffer = (byte *)HeapAlloc( GetProcessHeap(), 0, nCapacity );					
					#elif defined(__GNUC__)
					m_pBuffer = (byte *)malloc( nCapacity );					
					#else
					#	error No platform support here.
					#endif
				#endif

				assert(m_pBuffer);
				m_nSize = nCapacity;
			}
		}

		inline void Buffer::Realloc(UInt64 nSize)
		{
			if (nSize > UInt32_MaxValue) throw NotImplementedException();

			if (nSize == 0) Free();
			else
			{
				if (m_nSize == nSize) return;

				#if 0
				if (m_nSize < nSize) return;
				if (!m_pBuffer) m_pBuffer = new byte [(UInt32)nSize];
				else
				{
					byte* pTmp = new byte [(UInt32)nSize];
					MoveMemory (pTmp, m_pBuffer, m_nSize);
					delete[] (byte*)m_pBuffer;
					m_pBuffer = pTmp;					
				}
				#elif 1
				if (!m_pBuffer) m_pBuffer = (byte *)malloc((UInt32) nSize);
				else m_pBuffer = (byte *)realloc( m_pBuffer, (UInt32)nSize);
				#else
					#if defined(_WINDOWS)
					if (!m_pBuffer) m_pBuffer = (byte *)HeapAlloc( GetProcessHeap(), 0, nCapacity );
					else m_pBuffer = (byte *)HeapReAlloc( GetProcessHeap(), 0, m_pBuffer, nCapacity );
					#elif defined(__GNUC__)
					if (!m_pBuffer) m_pBuffer = (byte *)malloc( nCapacity );
					else m_pBuffer = (byte *)realloc( m_pBuffer, nCapacity );
					#else
					#	error No platform support here.
					#endif
				#endif

				assert(m_pBuffer);
				m_nSize = nSize;
			}
		}
	}
}

#endif	// __WBBuffer_h__

//	End of Buffer.h
