/*	utility.h
	Copyright (C) 2021 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __wbUtility_h__
#define __wbUtility_h__

#if !defined(EmulateSTL)
#include <utility>
#else
namespace std
{
	/** This may be required for some older version of Visual C++ or GNU, but I'm not sure which ones.  It also
		conflicts in some newer Visual Studio versions as it is part of an internal header that seems to be included
		automatically even without STL.  Until I identify a version that needs it, I'm disabling.  **/
	template <class T> struct remove_reference_t { typedef T type; };
	template <class T> struct remove_reference_t<T&> { typedef T type; };
	template <class T> struct remove_reference_t<T&&> { typedef T type; };

	template<class T> inline typename remove_reference_t<T>::type&& move(T&& _Arg)
	{
		return ((typename remove_reference_t<T>::type&&)_Arg);
	}

	template <class T> inline T&& forward(typename remove_reference_t<T>::type& _Arg)
	{
		return (static_cast<T&&>(_Arg));
	}

	template <class T> inline T&& forward(typename remove_reference_t<T>::type&& _Arg)
	{
		return (static_cast<T&&>(_Arg));
	}	
	
	// FUNCTION TEMPLATE move
	template <class _Ty>
	constexpr remove_reference_t<_Ty>&& move(_Ty&& _Arg) noexcept { // forward _Arg as movable
		return static_cast<remove_reference_t<_Ty>&&>(_Arg);
	}

	template<class T> inline void swap(T& a, T& b)
	{
		T tmp = std::move(a); a = std::move(b); b = std::move(tmp);
	}
}

#endif	// EmulateSTL

#endif	// __wbUtility_h__

//	End of utility.h
