/////////
//	Common.h
//	Copyright (C) 1998-2014 by Wiley Black
//

/** If header files need to enforce a certain sequencing, they can include Common.h ahead of their #pragma once or #ifndef block.
	Common.h must not include anything outside of its __Common_h__ once-exclusion block to facilitate this, because Common.h
	is allowed as the first header file in any sequence. **/

#ifndef __wbCommon_h__
#define __wbCommon_h__

#include "wbConfiguration.h"

/** Attempt to determine architecture characteristics, if available... **/
/** The caller can also defined LITTLE_ENDIAN or BIG_ENDIAN before including the library.  This provides an optimization. **/
#if defined(i386) || defined(__i686__) || defined(_M_IX86) || defined(_X86_) || defined(_X86)
#define LITTLE_ENDIAN
#ifndef _X86
#	define _X86
#endif
#endif

#if defined(_X64_) || defined(_M_X64) || defined(_M_AMD64) || defined(_X64)
#define LITTLE_ENDIAN
#ifndef _X64
#	define _X64
#endif
#endif

namespace wb
{
	template<typename T> inline T MaxOf(T a, T b) { return (a > b) ? (a) : (b); }
	template<typename T> inline T MinOf(T a, T b) { return (a < b) ? (a) : (b); }
	template<typename T> inline T MaxOf(T a, T b, T c) { return MaxOf(MaxOf(a,b),c); }
	template<typename T> inline T MinOf(T a, T b, T c) { return MinOf(MinOf(a,b),c); }

	template<typename T> inline int Round(T a) { 
		int flrA = (int)a;
		// Note: if getting compiler warnings here, consider whether T is a floating-point type.  If it isn't,
		// then Round() may be pointless.
		return (a >= (T)0.0)
			? ((a - (T)flrA) >= ((T)0.5) ? flrA+1 : flrA)
			: ((a - (T)flrA) >= ((T)-0.5) ? flrA : flrA-1);
	}
};

/** Exactly one .cpp file must include this header file with 'PrimaryModule' defined.  All other inclusions should leave
	Primary undefined. **/
#ifdef PrimaryModule
#define global_variable
#else
#define global_variable extern
#endif

#endif	// __wbCommon_h__

//	End of Common.h

