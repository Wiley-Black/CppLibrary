/////////
//  Language.h
//  Copyright (C) 1999-2002, 2014 by Wiley Black
/////////
//  Provides enhancements and conveniences within the C++ language.
/////////

#ifndef __WBLanguage_h__
#define __WBLanguage_h__

//#include "../wbFoundation.h"

/** AddFlagSupport(EnumType) is used to define the operators &, |, ^, ~, &=, |=, ^=, ==, and != for an enum class.  Once defined,
	this enables the use of the enum class in a "flag style" without typecasts.  For example:

	enum class MyFlags : UInt32
	{
		Alpha	= 0x1,
		Beta	= 0x2,
		Gamma	= 0x4
	}
	AddFlagSupport(MyFlags);

	void ExampleUse()
	{
		MyFlags value = MyFlags::Beta | MyFlags::Gamma;
		if (value & MyFlags::Gamma != 0) printf("Gamma present!\n");
	}

	Use of an embedded typename is supported.  For example, AddFlagSupport(MyClass::MyFlags) is allowed.
**/

/** Provide emulation via macro for strongly-type enums (enum class) when not supported **/

#if defined(GCC_VERSION) && (GCC_VERSION < 40400)	
#define enum_class_start(x,type)					\
		class x {										\
			typedef type ty;							\
			ty Value;									\
		public:											\
			enum Values
#define enum_class_end(x)	;						\
			x() { Value = 0; }							\
			x(const x& val) { Value = val.Value; }		\
			x(Values val) { Value = (ty)val; }			\
			explicit x(int val) { Value = (ty)val; }	\
			x& operator=(const x& val) { Value = val.Value; return *this; }		\
			x& operator=(Values val) { Value = (ty)val; return *this; }			\
			bool operator==(int b) const { return (ty)Value == (ty)b; }			\
			bool operator!=(int b) const { return (ty)Value != (ty)b; }			\
			bool operator==(const x& b) const { return Value == b.Value; }		\
			bool operator!=(const x& b) const { return Value != b.Value; }		\
			operator ty() const { return Value; }				\
		};		
	//#define enum_type(x)					x::Values
#define UsingEnumClassEmulation
#else
#define enum_class_start(x,type)		enum class x : type
#define enum_class_end(x)				
//#define enum_type(x)					x				// macro name conflicts with a google protobuf name.  Also doesn't seem to be used.
#endif

/** Provide AddFlagSupport() macro to allow enum class definitions to support flag behavior such as |, &, ^, and ~ operators. **/

#if defined(GCC_VERSION) && (GCC_VERSION < 40400)	

	/** We're using emulated strongly-typed enum support (enum class) via the macros provided above.  The emulation
		is less restrictive than the C++ standard, and automatic conversion to integral types seems to cover it. **/

#define AddFlagSupport(EnumType)				\
	inline EnumType& operator&=(EnumType& x, int y) { x = static_cast<EnumType>(static_cast<static_cast<std::underlying_type<EnumType>::type>(x) & static_cast<static_cast<std::underlying_type<EnumType>::type>(y)); return x; }		\
	inline EnumType& operator|=(EnumType& x, int y) { x = static_cast<EnumType>(static_cast<static_cast<std::underlying_type<EnumType>::type>(x) | static_cast<static_cast<std::underlying_type<EnumType>::type>(y)); return x; }		\
	inline EnumType& operator^=(EnumType& x, int y) { x = static_cast<EnumType>(static_cast<static_cast<std::underlying_type<EnumType>::type>(x) ^ static_cast<static_cast<std::underlying_type<EnumType>::type>(y)); return x; }	

#else

	/** Using language's native strongly-typed enum support **/
#define AddFlagSupport(EnumType)			\
	inline EnumType	operator&(EnumType x, EnumType y) { return static_cast<EnumType>(static_cast<std::underlying_type<EnumType>::type>(x) & static_cast<std::underlying_type<EnumType>::type>(y)); }	\
	inline EnumType	operator|(EnumType x, EnumType y) { return static_cast<EnumType>(static_cast<std::underlying_type<EnumType>::type>(x) | static_cast<std::underlying_type<EnumType>::type>(y)); }	\
	inline EnumType	operator^(EnumType x, EnumType y) { return static_cast<EnumType>(static_cast<std::underlying_type<EnumType>::type>(x) ^ static_cast<std::underlying_type<EnumType>::type>(y)); }	\
	inline EnumType	operator~(EnumType x) { return static_cast<EnumType>(~static_cast<std::underlying_type<EnumType>::type>(x)); }	\
	inline EnumType& operator&=(EnumType& x, EnumType y) { x = x & y; return x; }		\
	inline EnumType& operator|=(EnumType& x, EnumType y) { x = x | y; return x; }		\
	inline EnumType& operator^=(EnumType& x, EnumType y) { x = x ^ y; return x; }		\
	inline bool	operator!(EnumType x) { return static_cast<std::underlying_type<EnumType>::type>(x) == 0; }	\
	inline bool operator==(const EnumType& x, std::underlying_type<EnumType>::type y) { return static_cast<std::underlying_type<EnumType>::type>(x) == y; }		\
	inline bool operator!=(const EnumType& x, std::underlying_type<EnumType>::type y) { return static_cast<std::underlying_type<EnumType>::type>(x) != y; }		\
	inline bool any(EnumType x) { return static_cast<std::underlying_type<EnumType>::type>(x) != 0; }

#endif

/** Emulate log2() for Visual C++ versions below 2013 **/

#if defined(_MSC_VER) && _MSC_VER < 1800
#include <Math.h>

#undef log2
inline double log2(double n) { return log(n) / log((double)2.0); }
#endif

#endif	// __WBLanguage_h__

//  End of Language.h
