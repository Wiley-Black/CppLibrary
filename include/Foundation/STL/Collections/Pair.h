/*	Pair.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBPair_h__
#define __WBPair_h__

#include "../../../Platforms/Platforms.h"
#include "../../Exceptions.h"

#ifndef EmulateSTL

#include <utility>

/*
namespace wb
{	
	template <class T1, class T2> struct pair : public std::pair<T1,T2> 
	{ 
		typedef std::pair<T1,T2> base;
	public:		
		pair() : base() { }
		template<class U, class V> pair (const pair<U,V>& pr) : base(pr) { }
		pair (const typename base::first_type& a, const typename base::second_type& b) : base(a,b) { }
	};

	template< class T1, class T2 > wb::pair<T1,T2> make_pair( T1&& t, T2&& u );
}
*/

#else

namespace std
{
	template <class T1, class T2> struct pair
	{
		typedef T1 first_type;
		typedef T2 second_type;

		first_type first;
		second_type second;

		/** Constructors **/
		
		pair() : first(), second() { }

		template<class U, class V> pair (const pair<U,V>& pr)
			: first(pr.first), second(pr.second)
		{			
		}

		template<class U, class V> pair (pair<U,V>&& pr)
			: first(wb::move(pr.first)), second(wb::move(pr.second))
		{
		}

		pair (const pair& pr) 
			: first(pr.first), second(pr.second)
		{			
		}

		pair (pair&& pr)
			: first(wb::move(pr.first)), second(wb::move(pr.second))
		{
		}

		pair (const first_type& a, const second_type& b)
			: first(a), second(b)
		{
		}

#		if 0	// Not sure how to implement this offhand...
		template<class U, class V> pair (U&& a, V&& b)
			first(a), second(b)
		{
		}

		template <class... Args1, class... Args2>
		  pair (piecewise_construct_t pwc, tuple<Args1...> first_args,
										   tuple<Args2...> second_args);
#		endif

		/** operator= **/
		  
		pair& operator= (const pair& pr) { first = pr.first; second = pr.second; return *this; }
		template <class U, class V> pair& operator= (const pair<U,V>& pr) { first = pr.first; second = pr.second; return *this; }
		pair& operator= (pair&& pr) { first = std::move(pr.first); second = std::move(pr.second); return *this; }
		template <class U, class V> pair& operator= (pair<U,V>&& pr) { first = std::move(pr.first); second = std::move(pr.second); return *this; }
	};
}

#	endif

#endif	// __WBPair_h__

//	End of WBPair.h

