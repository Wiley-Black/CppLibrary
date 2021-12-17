/////////
//	FixedRingBuffer.h
//	Copyright (C) 2001, 2014 by Wiley Black
/////////

#ifndef __FixedRingBuffer_h__
#define __FixedRingBuffer_h__

/////////
//	Dependencies
//

#include "Memory Management/FlexRingBufferEx.h"

namespace wb
{
	namespace memory
	{
		/** FixedRingBuffer **/

		class FixedRingBuffer : public FlexRingBufferEx
		{
		protected:
			virtual uint GetInitialAllocate() const { return GetAlloc(); }
			virtual bool OnAllocate(uint dwNewAlloc);
			virtual uint GetMaximumAllocate() const { return GetAlloc(); }

		public:			
			FixedRingBuffer(uint nBufferSize);
		};

		/** Implementation **/

		inline bool FixedRingBuffer::OnAllocate(uint dwNewAlloc)
		{
			if (m_pBuffer) return false;
			return FlexRingBufferEx::OnAllocate(dwNewAlloc);
		}

		inline FixedRingBuffer::FixedRingBuffer(uint nBufferSize)
		{
			OnAllocate(nBufferSize);
		}
	}
}

#endif	// __FixedRingBuffer_h__

//	End of FixedRingBuffer.h


