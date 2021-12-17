/*	cstddef.h
	Copyright (C) 2021 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __wbStlStdDef_h__
#define __wbStlStdDef_h__

#if defined(EmulateSTL) || ((defined(_MSC_VER) && _MSC_VER < 1600) || (defined(GCC_VERSION) && GCC_VERSION < 40600))
#if ((defined(_MSC_VER) && _MSC_VER < 1600) || (defined(GCC_VERSION) && GCC_VERSION < 40600))
#define nullptr	(0)

// The following definition may have to go under different versioning testing than the above definition, but it can also 
// conflict with the built-in definition of nullptr_t that sometimes comes up.  May need fine tuning for older version support.
namespace std { typedef void* nullptr_t; }			
#endif
#else
#include <cstddef>			// For nullptr_t
#endif

#endif	// __wbStlStdDef_h__

//	End of cstddef.h
