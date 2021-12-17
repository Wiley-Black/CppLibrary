/////////
//	HistoryRingBuffer.h
//	Copyright (C) 2001, 2014 by Wiley Black
/////////
//	Similar to a FixedRingBuffer, but efficiently designed for a single purpose - maintaining
//	a history buffer of a fixed ("maximum history") length.
/////////

#ifndef __HistoryRingBuffer_h__
#define __HistoryRingBuffer_h__

namespace wb
{
	namespace memory
	{
		/// <summary>
		/// The HistoryRingBuffer buffer provides a fixed-size memory window of recent data.  Although this 
		/// could (maybe should) be based on the RingBuffer classes, for simplicity and efficiency it is 
		/// implemented independently here.
		/// </summary>
		template<class T> class HistoryRingBuffer
		{
			T* Buffer;
			int Capacity;			// Allocated size of buffer.
			int iHead;				// Data is read from the head, and then removed from the ring.
			int iTail;			    // Data is added at the tail, and the length increases.

		public:
			HistoryRingBuffer(int nElements)
			{
				Buffer = new T[nElements+1];        // One entry is lost by the case where iHead = iTail.
				Capacity = nElements+1;
				iHead = iTail = 0;
			}

			~HistoryRingBuffer()
			{
				delete[] Buffer;
				Buffer = nullptr;
			}

			int length()
			{					
				if (iTail < iHead) return (Capacity - iHead) + iTail;			// Wraps-around
				else return iTail - iHead;									    // Linear or empty					
			}

			void Add(T obj)
			{
				Buffer[iTail++] = obj;
				if (iTail >= Capacity) iTail = 0;
				if (iTail == iHead) 
				{
					// This happens when the buffer is full and we have just overwritten the
					// oldest entry.
					iHead++;
					if (iHead >= Capacity) iHead = 0;
				}
			}

			void Add(T* obj, int count)
			{
				// TODO: Optimization possible here.
				int offset = 0;
				while (count > 0) { Add(obj[offset++]); count--; }            
			}

			/// <summary>
			/// GetHistory() retrieves the Nth-from-last element in the ring buffer.  For example,
			/// calling Add(1), Add(2), and Add(3) on a new buffer results in the buffer containing
			/// the entries 1, 2, and 3.  Calling GetHistory(1) returns 3.  Calling GetHistory(3)
			/// returns 1.  Calling GetHistory(0) or with a distance greater than the buffer's
			/// length result in an exception.
			/// </summary>
			/// <param name="nDistance">Distance in history of element to retrieve.  Must be greater
			/// than zero and less than or equal to Length.</param>
			/// <returns>The value of the element.</returns>
			T GetHistory(int nDistance)
			{
				if (nDistance <= 0 || nDistance > length()) throw ArgumentOutOfRangeException();            
				int ii = iTail - nDistance;
				if (ii < 0) ii += Capacity;
				return Buffer[ii];
			}
		};
	}
}

#endif	// __HistoryRingBuffer_h__

//	End of HistoryRingBuffer.h

