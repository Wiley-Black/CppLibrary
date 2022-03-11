/////////
//  Queue.h
//  Copyright (C) 2022 by Wiley Black
/////////

#ifndef __WBQueue_h__
#define __WBQueue_h__

#include "wbFoundation.h"

#ifndef EmulateSTL
#include <queue>

#if 0
namespace wb
{
	//namespace collections
	// {
		//using namespace std;

		template <class T, class Container = std::deque<T> > class stack : public std::stack<T,Container>
		{
			typedef std::stack<T,Container> base;
		public:
		};
	// }
}
#endif

#else

#include "Vector.h"

namespace std
{			
	template<class T, class Container = vector<T>> class queue
	{
	};
}

#endif	// EmulateSTL

#endif	// __WBQueue_h__

//  End of Queue.h
