/////////
//  ThreadSafeQueue.h
//  Copyright (C) 2024 by Wiley Black
/////////
//  Provides the ThreadSafeQueue class, which is a convenient thread-safe
//  wrapper around the STL queue class.
/////////

#ifndef __WBThreadSafeQueue_h__
#define __WBThreadSafeQueue_h__

#include "../wbFoundation.h"
#include "../Foundation/STL/Collections/Queue.h"
#include "../System/Monitor.h"

namespace wb
{	
	template<typename T> class ThreadSafeQueue
	{
		std::queue<T>	m_queue;
		CriticalSection	m_cs;

	public:		
		typedef typename std::queue<T>::size_type size_type;

		ThreadSafeQueue() {}				

		void push(T&& Value) 
		{ 
			wb::Lock lock(m_cs);
			m_queue.push(std::move(Value));
		}

		bool try_pop(T& result)
		{
			wb::Lock lock(m_cs);
			if (m_queue.size() > 0)
			{
				result = std::move(m_queue.front());
				m_queue.pop();
				return true;
			}
			else
			{
				return false;
			}			
		}

		size_type size()
		{
			wb::Lock lock(m_cs);
			return m_queue.size();
		}
	};
}

#endif	// __ThreadSafeQueue_h__

//  End of ThreadSafeQueue.h
