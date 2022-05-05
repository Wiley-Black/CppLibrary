/*	Memory.h
	Copyright (C) 2021 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __wbStlMemory_h__
#define __wbStlMemory_h__

#include <malloc.h>
#include <memory>

#ifdef EmulateSTL
namespace std
{
	/** Default Allocator **/

	template<class T> class allocator
	{
	public:
		typedef T value_type;
		typedef T* pointer;
		typedef T& reference;
		typedef const T* const_pointer;
		typedef const T& const_reference;
		typedef size_t size_type;

		allocator() throw() { }
		allocator(const allocator& alloc) throw() { }
		template <class U> allocator(const allocator<U>& alloc) throw() { }
		~allocator() { }

		pointer allocate(size_type n);
		void deallocate(pointer p);
		pointer reallocate(pointer p, size_type new_count);
		void destroy(pointer p);
	};

	/** Void specialization **/

	template <> class allocator<void> {
	public:
		typedef void* pointer;
		typedef const void* const_pointer;
		typedef void value_type;
		// template <class U> struct rebind { typedef allocator<U> other; };
		typedef size_t size_type;

		pointer allocate(size_type n);
		void deallocate(pointer p);
		pointer reallocate(pointer p, size_type new_size);
		void destroy(pointer p);
	};

	/** Smart Pointers **/

	template<class T> struct default_delete
	{
		void operator()(T* ptr) const { if (ptr != nullptr) delete ptr; }
	};

	template<class T> struct default_delete<T[]>
	{
		void operator()(T* ptr) const { if (ptr != nullptr) delete[] ptr; }
	};

	template<class T, class Deleter = default_delete<T> > class unique_ptr
	{
		unique_ptr(const unique_ptr&);					// Not defined.
		unique_ptr& operator= (const unique_ptr&);		// Not defined.

	public:
		typedef T element_type;
		typedef T* pointer;

	private:
		pointer		m_p;
		Deleter		m_del;

		void DoDelete()
		{
			if (m_p != nullptr) get_deleter()(m_p);
		}

	public:
		constexpr_please unique_ptr() : m_p(nullptr) { }
		constexpr_please unique_ptr(nullptr_t ptr) : m_p((pointer)ptr) { }
		explicit unique_ptr(pointer p) : m_p(p) { }
		unique_ptr(unique_ptr&& u) { m_p = u.m_p; u.m_p = nullptr; m_del = u.m_del; }
		template< class U, class E > unique_ptr(unique_ptr<U, E>&& u) { m_p = u.m_p; u.m_p = nullptr; m_del = u.m_del; }
		~unique_ptr() { DoDelete(); }

		unique_ptr& operator= (unique_ptr&& x) noexcept {
			DoDelete();
			m_p = x.m_p; x.m_p = nullptr; m_del = x.m_del;
			return *this;
		}
		unique_ptr& operator= (nullptr_t ptr) noexcept {
			DoDelete();
			m_p = ptr;
			return *this;
		}
		template <class U, class E> unique_ptr& operator= (unique_ptr<U, E>&& x) noexcept {
			DoDelete();
			m_p = x.m_p; x.m_p = nullptr; m_del = x.m_del;
			return *this;
		}

		pointer get() const noexcept { return m_p; }
		pointer release() noexcept { pointer tmp = m_p; m_p = nullptr; return tmp; }
		// explicit operator bool() const noexcept { return m_p != nullptr; }
		bool IsNull() const { return m_p == nullptr; }

		element_type& operator*() const { return *m_p; }
		pointer operator->() const { return m_p; }

		Deleter& get_deleter() { return m_del; }
		const Deleter& get_deleter() const { return m_del; }
	};

	template <class T, class Deleter> class unique_ptr<T[], Deleter>
	{
		unique_ptr(const unique_ptr& cp);		// Not defined.
	public:
		typedef T element_type;
		typedef T* pointer;

	private:
		pointer		m_p;
		Deleter		m_del;

		void DoDelete()
		{
			if (m_p != nullptr) get_deleter()(m_p);
		}

	public:
		constexpr_please unique_ptr() : m_p(nullptr) { }
		constexpr_please unique_ptr(nullptr_t ptr) : m_p((pointer)ptr) { }
		explicit unique_ptr(pointer p) : m_p(p) { }
		unique_ptr(unique_ptr&& u) { m_p = u.m_p; u.m_p = nullptr; m_del = u.m_del; }
		template< class U, class E > unique_ptr(unique_ptr<U, E>&& u) { m_p = u.m_p; u.m_p = nullptr; m_del = u.m_del; }
		~unique_ptr() { DoDelete(); }

		unique_ptr& operator= (unique_ptr&& x) noexcept {
			DoDelete();
			m_p = x.m_p; x.m_p = nullptr; m_del = x.m_del;
			return *this;
		}
		unique_ptr& operator= (nullptr_t ptr) noexcept {
			DoDelete();
			m_p = ptr;
			return *this;
		}
		template <class U, class E> unique_ptr& operator= (unique_ptr<U, E>&& x) noexcept {
			DoDelete();
			m_p = x.m_p; x.m_p = nullptr; m_del = x.m_del;
			return *this;
		}

		pointer get() const noexcept { return m_p; }
		pointer release() noexcept { pointer tmp = m_p; m_p = nullptr; return tmp; }
		//explicit operator bool() const noexcept { return m_p != nullptr; }
		bool IsNull() const { return m_p == nullptr; }

		element_type& operator*() const { return *m_p; }
		pointer operator->() const { return m_p; }

		Deleter& get_deleter() { return m_del; }
		const Deleter& get_deleter() const { return m_del; }

		element_type& operator[](size_t i) const { return *(&m_p[i]); }	  // only defined in array-specialization
	};

	/** Implementations **/

	template<class T> inline T* allocator<T>::allocate(size_type n)
	{
		return (pointer)malloc(n);
	}

	template<class T> inline void allocator<T>::deallocate(T* p)
	{
		if (p) free(p);
	}

	template<class T> inline T* allocator<T>::reallocate(T* p, size_type new_count)
	{
		return (pointer)realloc(p, new_size);
	}

	template<class T> inline void allocator<T>::destroy(T* p)
	{
		p->~value_type();
	}

	inline void* allocator<void>::allocate(size_type n)
	{
		return (pointer)malloc(n);
	}

	inline void allocator<void>::deallocate(void* p)
	{
		if (p) free(p);
	}

	inline void* allocator<void>::reallocate(void* p, size_type new_size)
	{
		return realloc(p, new_size);
	}

	inline void allocator<void>::destroy(void* p)
	{
	}
}
#endif

#endif	// __wbStlMemory_h__

//	End of Memory.h
