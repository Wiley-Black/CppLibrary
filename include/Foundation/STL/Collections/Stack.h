/////////
//  Stack.h
//  Copyright (C) 2014 by Wiley Black
/////////

#ifndef __WBStack_h__
#define __WBStack_h__

#include "wbFoundation.h"

#ifndef EmulateSTL
#include <stack>

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
	template<class T, class Container = vector<T>> class stack
	{
	};
}

#endif	// EmulateSTL

#endif	// __WBStack_h__

//  End of Stack.h
