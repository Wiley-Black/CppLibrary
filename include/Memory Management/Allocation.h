/*	Allocation.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WbAllocation_h__
#define __WbAllocation_h__

#include "../Platforms/Platforms.h"
#include "../Foundation/STL/Memory.h"
#include <malloc.h>

namespace wb
{
	template<class T> class allocator : public std::allocator<T> { };			

	namespace memory
	{
		/// <summary>Responsibility-Pointer (r_ptr) is very similar to the unique_ptr class offered by STL, but can accept and 
		/// wrap a pointer with or without taking responsibility for freeing its memory.  An r_ptr object can only be 
		/// moved to another pointer, not copied.  Initialization of an r_ptr requires use of either the responsible() or
		/// absolved() call.  The use patterns for r_ptr look like:
		/// <example>
		///	r_ptr&lt;Object&gt; owned_object_ptr = r_ptr&lt;Object&gt;::responsible(new Object(...));		// Initializes object_ptr and takes responsibility for deletion.
		///
		/// Object* pGivenPtr = ...;
		/// r_ptr&lt;Object&gt; object_ptr = r_ptr&lt;Object&gt;::absolved(pGivenPtr);						// Initializes object_ptr but does not assume responsibility for deletion.
		///
		///	ToAnotherFunction(std::move(object_ptr));														// Calls a function taking a single r_ptr&lt;Object&gt;&& parameter.  Transfers object_ptr to that function.
		///	// object_ptr has now been moved into another r_ptr and use here would result in a null pointer exception.
		///
		/// ToAnotherFunction(r_ptr&lt;Object&gt;::absolved(owned_object_ptr));								// Passes an absolved copy of owned_object_ptr to a function.  
		/// // owned_object_ptr is still valid and deletion of the Object will only occur after owned_object_ptr passes out of scope.
		/// </example>
		///	</summary>
		template<typename T, class AllocatorType = allocator<T>> class r_ptr
		{
			typedef T*		pointer;
			typedef T&		reference;
			pointer			m_pObj;
			bool			m_bResponsible;
			AllocatorType	TheAllocator;

		private:

			r_ptr(pointer p, bool bResponsible) { m_pObj = p; m_bResponsible = bResponsible; }

		public:

			/** Explicit forms **/

			// To create an r_ptr that will automatically call delete on the pointer or reference given...
			static r_ptr responsible(pointer p) { return r_ptr(p, true); }
			static r_ptr responsible(reference obj) { return r_ptr(&obj, true); }

			// To create an r_ptr that will not delete the object given...
			static r_ptr absolved(pointer p) { return r_ptr(p, false); }
			static r_ptr absolved(reference obj) { return r_ptr(&obj, false); }
			static r_ptr absolved(const r_ptr& obj) { return r_ptr(obj.m_pObj, false); }

			/** Initialization **/

			/*
			r_ptr() { m_pObj = nullptr; m_bResponsible = false; }
			// constexpr r_ptr (nullptr_t p) { m_pObj = nullptr; m_bResponsible = false; }
			explicit r_ptr (pointer p) { m_pObj = p; m_bResponsible = true; }
			explicit r_ptr (reference obj) { m_pObj = &obj; m_bResponsible = false; }
			*/

			r_ptr ()
			{
				m_pObj = nullptr; m_bResponsible = false;
			}

			r_ptr (std::nullptr_t)
			{
				m_pObj = nullptr; m_bResponsible = false;
			}

			r_ptr (r_ptr&& x) 
			{
				m_pObj = x.m_pObj; x.m_pObj = nullptr;
				m_bResponsible = x.m_bResponsible; x.m_bResponsible = false;
			}

			r_ptr& operator= (std::nullptr_t)
			{ 
				if (m_bResponsible && m_pObj != nullptr) TheAllocator.destroy(m_pObj); //delete m_pObj;
				m_pObj = nullptr; 
				m_bResponsible = false; 
				return *this;
			}

			r_ptr& operator= (r_ptr&& x) 
			{ 
				if (m_bResponsible && m_pObj != nullptr) TheAllocator.destroy(m_pObj); //delete m_pObj;
				m_pObj = x.m_pObj; x.m_pObj = nullptr;
				m_bResponsible = x.m_bResponsible; x.m_bResponsible = false;
				return *this;
			}
			/*
			r_ptr& operator= (pointer p) { 		
				if (m_bResponsible && m_pObj != nullptr) TheAllocator.destroy(m_pObj); // delete m_pObj;
				m_pObj = p;			// p can be nullptr since "delete null" has no effect.
				m_bResponsible = true;
				return *this;
			}
			r_ptr& operator= (reference p) { 
				if (m_bResponsible && m_pObj != nullptr) TheAllocator.destroy(m_pObj); // delete m_pObj;
				m_pObj = &p;			// p can be nullptr since "delete null" has no effect.
				m_bResponsible = false;
				return *this;
			}
			*/
			bool operator== (pointer p) const { return (m_pObj == p); }
			bool operator!= (pointer p) const { return (m_pObj != p); }
			bool operator== (const r_ptr<T>& p) const { return (m_pObj == p.m_pObj); }
			bool operator!= (const r_ptr<T>& p) const { return (m_pObj != p.m_pObj); }

			/// <summary>Release a pointer that this object has responsibility for.  Cannot be used
			///	when the r_ptr is not responsible.</summary>
			pointer release()
			{
				assert (m_bResponsible);
				m_bResponsible = false;
				return m_pObj;
			}

			bool IsResponsible() const { return m_bResponsible; }

			/** Cleanup **/

			~r_ptr()
			{
				if (m_bResponsible && m_pObj != nullptr) TheAllocator.destroy(m_pObj); // delete m_pObj;
				m_pObj = nullptr;
				m_bResponsible = false;
			}

			/** Accessors **/

			pointer get() const { return m_pObj; }
			//operator bool() const { return (m_pObj != nullptr); }
			bool IsAssigned() const { return (m_pObj != nullptr); }
			pointer operator->() const { return m_pObj; }
			reference operator*() const { return *m_pObj; }
		};
	}

	/** Implementations **/
}

#endif	// __WbAllocation_h__

//	End of Allocation.h
