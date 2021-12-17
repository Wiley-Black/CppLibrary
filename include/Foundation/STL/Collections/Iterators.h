/*	Iterators.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBIterators_h__
#define __WBIterators_h__

#include "../../../Platforms/Platforms.h"
#include "../../Exceptions.h"
#include "../Text/String.h"
#include "Pair.h"

#ifndef EmulateSTL
#include <iterator>
#else

namespace std
{
	#if defined(_X86)
	typedef Int32 ptrdiff_t;
	#elif defined(_X64)
	typedef Int64 ptrdiff_t;
	#endif

	struct input_iterator_tag {};
	struct output_iterator_tag {};
	struct forward_iterator_tag {};
	struct bidirectional_iterator_tag {};
	struct random_access_iterator_tag {};

	template <class Category, class T, class Distance = ptrdiff_t, class Pointer = T*, class Reference = T&>
	struct iterator {
		typedef T         value_type;
		typedef Distance  difference_type;
		typedef Pointer   pointer;
		typedef Reference reference;
		typedef Category  iterator_category;
	};
}

#	endif

#endif	// __WBIterators_h__

//	End of Iterators.h

