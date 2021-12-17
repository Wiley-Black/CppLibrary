/*	Platform.h
	Copyright (C) 2014-2021 by Wiley Black (TheWiley@gmail.com)
	
	Provides macros and routines that make for more cross-platform programming and consistency.  For example, 
	the CopyMemory() macro is available on Windows but after including this header, it is available on any 
	platform.  If compiling on Windows, the existing CopyMemory() macro is used, but on GCC, it is defined.
*/

#ifndef __WbPlatform_h__
#define __WbPlatform_h__

#include "../Foundation/Common.h"

#include <math.h>
#include <limits.h>
#if (defined(_MSC_VER))
#include <tchar.h>
typedef unsigned long uint;
#endif
#if (defined(__GNUC__))
#include <sys/types.h>			// For size_t.  Also defines uint.
#endif
#include <float.h>				// For DBL_MIN and DBL_MAX.

typedef unsigned char       byte;
typedef unsigned char       UInt8;
typedef unsigned short      UInt16;
typedef unsigned int        UInt32;
typedef unsigned long long  UInt64;

typedef signed char         Int8;
typedef signed short        Int16;
typedef signed int          Int32;
typedef signed long long    Int64;

#define UInt8_MinValue		0u
#define UInt16_MinValue		0u
#define UInt32_MinValue		0ul
#define UInt64_MinValue		0ull

#define UInt8_MaxValue		UCHAR_MAX
#define UInt16_MaxValue		USHRT_MAX
#define UInt32_MaxValue		0xFFFFFFFF
#define UInt64_MaxValue		ULLONG_MAX

#define Int8_MinValue		SCHAR_MIN
#define Int16_MinValue		SHRT_MIN
#define Int32_MinValue		(-2147483647-1)
#define Int64_MinValue		LLONG_MIN

#define Int8_MaxValue		SCHAR_MAX
#define Int16_MaxValue		SHRT_MAX
#define Int32_MaxValue		2147483647
#define Int64_MaxValue		LLONG_MAX

#define Float_MaxValue		FLT_MAX
#define Double_MaxValue		DBL_MAX

#define Float_MinValue		FLT_MIN
#define Double_MinValue		DBL_MIN

static const size_t size_t_MaxValue = (size_t)-1;

#if defined(__GNUC__)
	// Define a single macro to calculate GNUC version.  For example, GCC 3.2.0 will yield GCC_VERSION 30200.
#define GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
#endif

#if (defined(_MSC_VER) && (_MSC_VER > 1800)) || (defined(GCC_VERSION) && (GCC_VERSION >= 40600))
	// Not sure on exact versions.  Also there is confusion between whether the compiler truly supports
	// constexpr or at least recognizes the keyword without error (which is the goal here).  The constexpr_please
	// macro uses constexpr if the compiler supports it, or evaluates to nothing when not supported.  Due to
	// compiler recognition with poor support, but the limitation that the C++ standard forbids applying macros
	// to recognized keywords, the base keyword can't be used.
#define constexpr_please	constexpr
#else
#define constexpr_please
#endif

#if (defined(_MSC_VER) && (_MSC_VER < 1700)) || (defined(GCC_VERSION) && (GCC_VERSION < 40600))
#define noexcept
#endif

#if defined(_WIN32) || defined(_WIN64)
#ifndef _WINDOWS
#define _WINDOWS
#endif
#else
#define _LINUX
#endif

#if defined(PrimaryModule) || defined(DmlPrimaryModule)
#if defined(EmulateSTL) && defined(_WINDOWS) && defined(_MSC_VER)
	#if defined(UNICODE)
		#pragma message("Compiling using Visual C++ in Windows with STL emulation, UNICODE enabled.")
	#else
		#pragma message("Compiling using Visual C++ in Windows with STL emulation.")
	#endif
#elif !defined(EmulateSTL) && defined(_WINDOWS) && defined(_MSC_VER)
	#if defined(UNICODE)
		#pragma message("Compiling using Visual C++ in Windows with UNICODE enabled.")
	#else
		#pragma message("Compiling using Visual C++ in Windows.")
	#endif
#endif
#endif

#ifdef GCC_VERSION
    #define MaybeUnused __attribute__((used))
#else
	#define MaybeUnused
#endif

/** Verify minimum requirements **/

#if defined(GCC_VERSION)
	#if !defined(__cplusplus) || (__cplusplus < 201100L)
		#error The C++11 standard is required for this library.  For some compilers, you may need to enable it via a command-line option.
	#endif
#endif

/** Compatibility with older compiler and standards **/

#if (defined(GCC_VERSION) && GCC_VERSION < 40700)
#define override 
#endif

#if defined(_LINUX)
#	include <endian.h>
#	define htonll(x) htobe64(x)
#	define ntohll(x) be64toh(x)
#elif defined(_FreeBSD) || defined(_NetBSD)
#	include <sys/endian.h>
#	define htonll(x) htobe64(x)
#	define ntohll(x) be64toh(x)
#elif defined(_WINDOWS)
	// htonll() provided natively.
#else
#	error Platform support required.
#endif

namespace wb
{
	inline UInt16 SwapEndian(UInt16 val) { return (val << 8) | (val >> 8); }
	inline UInt32 SwapEndian(UInt32 val) { return (val << 24) | ((val <<  8) & 0x00ff0000) | ((val >>  8) & 0x0000ff00) | ((val >> 24) & 0x000000ff); }
	inline UInt64 SwapEndian(UInt64 val) { 
		return (val << 56) | ((val << 40) & 0x00ff000000000000ull) | ((val << 24) & 0x0000ff0000000000ull) | ((val << 8) & 0x000000ff00000000ull)
			| ((val >> 8) & 0x00000000ff000000) | ((val >> 24) & 0x0000000000ff0000ull) | ((val >> 40) & 0x000000000000ff00ull) | ((val >> 56) & 0x00000000000000ffull);
	}
	inline Int16 SwapEndian(Int16 val) { return SwapEndian((UInt16)val); }
	inline Int32 SwapEndian(Int32 val) { return SwapEndian((UInt32)val); }
	inline Int64 SwapEndian(Int64 val) { return SwapEndian((UInt64)val); }
	inline float SwapEndian(float val) { return (float)SwapEndian((UInt32)val); }
	inline double SwapEndian(double val) { return (double)SwapEndian((UInt64)val); }	

	inline Int32	Round32(double dVal){ return (fmod(dVal, 1.0) >= 0.5) ? (((Int32)dVal) + 1) : ((Int32)dVal); }
	inline Int64	Round64(double dVal){ return (fmod(dVal, 1.0) >= 0.5) ? (((Int64)dVal) + 1) : ((Int64)dVal); }	
}

	/** Provide these macros when not running on Windows **/
	
/** Additional platform support **/

#if !defined(_WINDOWS)
#if 0
#define MAKEWORD(lo, hi)	((WORD) (((BYTE) (lo)) | (((WORD) ((BYTE) (hi))) << 8)))
#define MAKELONG(lo, hi)	((LONG) (((WORD) (lo)) | (((DWORD) ((WORD) (hi))) << 16)))
#define HIBYTE(w)			((BYTE) (((WORD) (w) >> 8) & 0xFF))
#define LOBYTE(w)			((BYTE) (w))
#define HIWORD(l)			((WORD) (((DWORD) (l) >> 16) & 0xFFFF))
#define LOWORD(l)			((WORD) (l))
#endif
#endif	// !_WINDOWS

namespace wb
{
	inline Int64	MakeI64(Int32 lo, Int32 hi) { return ((Int64)lo) | (((Int64)hi) << 32ll); }
	inline UInt64	MakeU64(UInt32 lo, UInt32 hi) { return ((UInt64)lo) | (((UInt64)hi) << 32ull); }

	inline Int32	MakeI32(Int16 lo, Int16 hi) { return ((Int32)lo) | (((Int32)hi) << 16); }
	inline UInt32	MakeU32(UInt16 lo, UInt16 hi) { return ((UInt32)lo) | (((UInt32)hi) << 16); }

	inline Int32	MakeI32(Int8 lolo, Int8 lohi, Int8 hilo, Int8 hihi) {
		return ((Int32)lolo) | (((Int32)lohi) << 8) | (((Int32)hilo) << 16) | (((Int32)hihi) << 24);
	}
	inline UInt32	MakeU32(UInt8 lolo, UInt8 lohi, UInt8 hilo, UInt8 hihi) {
		return ((UInt32)lolo) | (((UInt32)lohi) << 8) | (((UInt32)hilo) << 16) | (((UInt32)hihi) << 24);
	}

	inline Int16	MakeI16(Int8 lo, Int8 hi) { return ((Int16)lo) | (((Int16)hi) << 8); }
	inline UInt16	MakeU16(UInt8 lo, UInt8 hi) { return ((UInt16)lo) | (((UInt16)hi) << 8); }

#if defined(LITTLE_ENDIAN)
	inline bool		IsLittleEndian() { return true; }
#elif defined(BIG_ENDIAN)
	inline bool		IsLittleEndian() { return false; }
#else
	inline bool		IsLittleEndian()
	{
		union { UInt32 i; char c[4]; } bint = { 0x01020304 };
		return bint.c[0] != 1;
	}
#endif
}

#if !defined(_WINDOWS)
    #define CopyMemory(Dest,Src,Len) memcpy(Dest,Src,(size_t)(Len))
    #define FillMemory(Dest,Len,Byte) memset(Dest,(int)Byte,(size_t)(Len))
    #define ZeroMemory(Dest,Len) memset(Dest,0,(size_t)(Len))
    #define MoveMemory(Dest,Src,Len) memmove(Dest,Src,(size_t)(Len))
#endif	// !_WINDOWS	

#endif	// __WbPlatform_h__

//	End of Platform.h
