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

/** unique_ptr helpers **/

#include "../Foundation/STL/Memory.h"
#include "../Foundation/STL/Collections/Stack.h"
#include "../Foundation/STL/Collections/Queue.h"

/// <summary>
/// is_type() is a helper function that tests whether the given pointer can be dynamic_cast to another pointer type.  
/// </summary>
/// <seealso>dynamic_pointer_movecast()</seealso>
/// <returns>True if the pointer can be cast to the specified type.  False if the pointer has a nullptr value or is 
/// not castable to the templated Target type as a pointer.</returns>
template<typename Target, typename Source> static inline bool is_type(const std::unique_ptr<Source>& ptr) {
	return dynamic_cast<Target*>(ptr.get()) != nullptr;
}

/// <summary>
/// is_type() is a helper function that tests whether the given pointer can be dynamic_cast to another pointer type.  
/// </summary>
/// <seealso>dynamic_pointer_movecast()</seealso>
/// <returns>True if the pointer can be cast to the specified type.  False if the pointer has a nullptr value or is 
/// not castable to the templated Target type as a pointer.</returns>
template<typename Target, typename Source> static inline bool is_type(const std::shared_ptr<Source>& ptr) {
	return dynamic_cast<Target*>(ptr.get()) != nullptr;
}

/// <summary>
/// dynamic_pointer_movecast() is a helper function that transfers from one unique_ptr to another unique_ptr while performing a dynamic typecast.  If the
/// original pointer is null or if the dynamic cast yields a nullptr (because the original pointer cannot be cast to the Target* type), then
/// the original object is deleted and a nullptr is returned.  In all cases, the original pointer is released.
/// </summary>
/// <example>
/// unique_ptr&lt;Derived> CreatePolymorphicType()
/// {
///		return make_unique&lt;Derived>();
/// }
/// 
/// ...
///	unique_ptr&lt;Base> pBase = CreatePolymorphicType();
/// if (is_type&lt;Derived>(pBase))
/// {
///		auto pDerived = dynamic_pointer_movecast(std::move(pBase));
///		assert (pBase == nullptr);
///		// Make use of Derived-specific features by pDerived pointer.
/// }
/// </example>
/// <seealso>is_type</seealso>
template<typename Target, typename Source> static inline std::unique_ptr<Target> dynamic_pointer_movecast(std::unique_ptr<Source>&& ptr) 
{
	Source* pTemp = ptr.release();
	if (pTemp == nullptr) return nullptr;
	
	if (dynamic_cast<Target*>(pTemp) == nullptr)
	{
		delete pTemp;
		return nullptr;
	}
	else return std::unique_ptr<Target>(dynamic_cast<Target*>(pTemp));
}

/// <summary>
/// Convenience function for popping the top of a unique_ptr stack.  This combines two operations: top() followed by
/// pop().  Since only one unique_ptr can hold the object, the top() call uses std::move() to extract the object's
/// pointer into a temporary unique_ptr, then pop() removes the element that now holds only nullptr.  The temporary
/// unique_ptr is then returned.
/// </summary>
/// <returns>Pointer to object that was previously on top of the stack.</returns>
template<typename T> static inline std::unique_ptr<T> pop_top(std::stack<std::unique_ptr<T>>& from_stack)
{
	auto ret = std::move(from_stack.top());
	from_stack.pop();
	return ret;
}

/// <summary>
/// Convenience function for popping the front of a unique_ptr queue.  This combines two operations: front() followed by
/// pop().  
/// </summary>
/// <seealso>pop_top()</seealso>
/// <returns>Pointer to object that was previously at the front of the queue.</returns>
template<typename T> static inline std::unique_ptr<T> pop_front(std::queue<std::unique_ptr<T>>& from_queue)
{
	auto ret = std::move(from_queue.front());
	from_queue.pop();
	return ret;
}

/// <summary>
/// Convenience function for popping the front of a unique_ptr vector.  This combines two operations: front() followed by
/// erase() at begin().  
/// </summary>
/// <seealso>pop_top()</seealso>
/// <returns>Pointer to object that was previously at the front of the queue.</returns>
template<typename T> static inline std::unique_ptr<T> pop_front(std::vector<std::unique_ptr<T>>& from)
{
	auto ret = std::move(from.front());
	from.erase(from.begin());
	return ret;
}

#endif	// __WBLanguage_h__

//  End of Language.h
