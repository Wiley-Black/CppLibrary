/*	UnorderedMap.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WbUnorderedMap_h__
#define __WbUnorderedMap_h__

#include "../../../Platforms/Platforms.h"
#include "../../Exceptions.h"
#include "../Text/String.h"
#include "Pair.h"
#include "Iterators.h"

#if !defined(EmulateSTL) && (!defined(GCC_VERSION) || GCC_VERSION > 40303)
	// It is not clear to me whether there is a definite issue with the unordered_map from 4.3.3,
	// however I am getting a compiler error that there is no at() member.  Rather than dig in for
	// confirmation, I am going to assume for now that this version simply lacks proper support.

#include <unordered_map>
#if (defined(_MSC_VER) && (_MSC_VER >= 1800)) || (defined(GCC_VERSION) && GCC_VERSION >= 40501)
	// Note: The exact version where initializer_list is well supported by GCC was not ascertained.  This
	// version was mentioned in Internet posts as being past the buggy phase.
	#include <initializer_list>
	#define InitializerListSupport
#endif

#if 0
namespace wb
{
	template < class Key,                                     // unordered_map::key_type
           class T,											// unordered_map::mapped_type
           class Hash = std::hash<Key>,                       // unordered_map::hasher
           class Pred = std::equal_to<Key>,                   // unordered_map::key_equal
           class Alloc = std::allocator< pair<const Key,T> >  // unordered_map::allocator_type
	> class unordered_map : public std::unordered_map<Key,T,Hash,Pred,Alloc> 
	{ 	
	public:
		typedef typename std::unordered_map<Key,T,Hash,Pred,Alloc> base;
		typedef typename std::unordered_map<Key,T,Hash,Pred,Alloc>::size_type size_type;
		typedef typename std::unordered_map<Key,T,Hash,Pred,Alloc>::iterator iterator;
		typedef typename std::unordered_map<Key,T,Hash,Pred,Alloc>::const_iterator const_iterator;
		typedef Key key_type;
		typedef T mapped_type;
		typedef typename std::pair<const key_type,mapped_type> value_type;	
		
		unordered_map() : base() { }
		unordered_map(size_type n) : base(n) { }
		unordered_map(const unordered_map& cp) : base(cp) { }
		unordered_map(unordered_map&& mv) : base(mv) { }

		unordered_map& operator=(const unordered_map& cp) { base::operator=(cp); return *this; }
		unordered_map& operator=(unordered_map&& mv) { base::operator=(mv); return *this; }
		
		/** wb version extensions **/

		void insert(const key_type& key, const mapped_type& value)
		{
			base::insert(value_type(key,value));
		}

		// Can't add our insert() extension in the derived class without providing the other overloads...
		std::pair<iterator,bool> insert ( const value_type& val ) { return base::insert(val); }
		template <class P> std::pair<iterator,bool> insert ( P&& val ) { return base::insert(val); }
		iterator insert ( const_iterator hint, const value_type& val ) { return base::insert(hint, val); }
		template <class P> iterator insert ( const_iterator hint, P&& val ) { return base::insert(hint, val); }
		// template <class InputIterator> void insert ( InputIterator first, InputIterator last ) { return base::insert(first, last); }
		#ifdef InitializerListSupport
		void insert ( std::initializer_list<value_type> il ) { base::insert(il); }
		#endif
	};
}
#endif

#else

namespace std
{		
	template <class T> class hash
	{
	public:
		UInt32 ComputeHash(T Value) const { return (UInt32)Value; }
	};

	template <> class hash <string>
	{
	public:
		UInt32 ComputeHash(string Value) const {
			UInt32 ret = 0;
			for (size_t ii=0; ii < Value.length(); ii++) ret += ((Value[ii] + 1) << 3) ^ ret;
			return ret;
		}
	};

	/// <summary>A reduced functionality version of the Standard Template Library unordered_map class.</summary>
	template <class Key, class T, class Hasher = hash<Key>> class unordered_map
	{	
	public:
		typedef Key key_type;
		typedef T mapped_type;
		//typedef pair<const Key, T> value_type;
			// STL uses the above with const, but that gives trouble with operator=() that I haven't
			// yet resolved.  This is functional, probably slight performance/type-safety hit...
		typedef pair<Key, T> value_type;

	private:
		typedef UInt32 hash_type;
		typedef size_t size_type;
		static const hash_type MaxHashBit = 0x80000000;		
		Hasher HashMaker;
		
		struct node
		{
			value_type	Value;
			node		*pNextNode;
		};
		
		hash_type	current_mask;
		node**		m_pBuckets;			// Array length is (current_mask+1)

		void Reallocate(size_type new_bucket_count)
		{
			assert (fmod(log2((double)new_bucket_count),1.0) < 1.0e-4);				// Must request a power-of-two for new_bucket_count.

				// To ask for 32 buckets, we need a mask of 11111b (0x1F, 31)...
			hash_type new_mask = (hash_type)(new_bucket_count - 1);
			if (new_mask == 0) new_mask = 1;
			if (new_mask <= current_mask) return;

			// Ensure that new_mask contains all 1's after the highest order bit (otherwise it won't function
			// properly as a mask).
			hash_type mask_bit = MaxHashBit;
			while (!(mask_bit & new_mask)) mask_bit >>= 1;
			while (mask_bit != 0) { new_mask |= mask_bit; mask_bit >>= 1; }

			node**	pOldBuckets = m_pBuckets;
			size_t	nOldBuckets = (current_mask + 1);			

			m_pBuckets = new node* [new_mask + 1];
			memset(m_pBuckets, 0, (new_mask + 1) * sizeof(node*));
			current_mask = new_mask;			

			// We have to use to an insert operation on each node as the bucket definitions are being revised..
			if (nOldBuckets > 1)
			{
				for (size_t ii=0; ii < nOldBuckets; ii++) 
				{
					node* pNode = pOldBuckets[ii];
					while (pNode != nullptr)
					{
						node* pNextNode = pNode->pNextNode;
						pNode->pNextNode = nullptr;
						reinsert(pNode);
						pNode = pNextNode;
					}
				}

				delete[] pOldBuckets;
			}
		}

		hash_type ComputeHash(const key_type& k) const { return HashMaker.ComputeHash(k); }

		static const size_t DefaultBucketCount = 32;

	public:				

		class iterator : public wb::iterator<wb::forward_iterator_tag,value_type>
		{
			// Past end will lead to an m_pLocal = nullptr situation, otherwise m_pLocal should have a value.
			const unordered_map<Key, T>&	m_Parent;
			size_type						m_Index;
			node*							m_pLocal;
		public:
			iterator(const unordered_map<Key, T>& Parent, size_type Index, node* pLocal) : m_Parent(Parent), m_Index(Index), m_pLocal(pLocal) { }
			iterator(const iterator& cp) : m_Parent(cp.m_Parent), m_Index(cp.m_Index), m_pLocal(cp.m_pLocal) { }			
			iterator& operator++(int) { return operator++(); }
			iterator& operator++() 
			{ 
				if (m_pLocal != nullptr) 
				{
					if (m_pLocal->pNextNode != nullptr) { m_pLocal = m_pLocal->pNextNode; return *this; }
					m_pLocal = nullptr;
				}

				m_Index ++;
				while (m_Index < m_Parent.bucket_count() && m_Parent.m_pBuckets[m_Index] == nullptr) m_Index ++;
				if (m_Index < m_Parent.bucket_count()) m_pLocal = m_Parent.m_pBuckets[m_Index];

				return *this; 
			}
			bool operator==(const iterator& rhs) { 
				if (m_pLocal == nullptr && rhs.m_pLocal == nullptr) return true;		// end iterators, no index needed.
				return m_Index == rhs.m_Index && m_pLocal == rhs.m_pLocal; 
			}
			bool operator!=(const iterator& rhs) { return !(operator==(rhs)); }
			value_type& operator*() 
			{ 
				if (m_pLocal == nullptr) throw IndexOutOfRangeException();				
				return m_pLocal->Value;
			}
			value_type* operator->() 
			{ 
				if (m_pLocal == nullptr) throw IndexOutOfRangeException();				
				return &(m_pLocal->Value);
			}
		};

		class local_iterator : public wb::iterator<wb::forward_iterator_tag,value_type>
		{
			node* p;
		public:
			local_iterator(node* x) :p(x) {}
			local_iterator(const local_iterator& mit) : p(mit.p) {}
			local_iterator& operator++() { p = p->pNextNode; return *this; }
			local_iterator& operator++(int) { p = p->pNextNode; return *this; }
			bool operator==(const local_iterator& rhs) {return p==rhs.p;}
			bool operator!=(const local_iterator& rhs) {return p!=rhs.p;}
			value_type& operator*() { return p->Value; }
			value_type* operator->() { return &(p->Value); }
		};

		unordered_map(size_type n = DefaultBucketCount)
			: end_iterator(*this, -1, nullptr),
			end_local_iterator(nullptr),
			m_pBuckets(nullptr),
			current_mask(0)
		{
			Reallocate(n);
		}

		unordered_map(const unordered_map& cp)
			: end_iterator(*this, -1, nullptr),
			end_local_iterator(nullptr),
			m_pBuckets(nullptr),
			current_mask(0)
		{
			Reallocate(cp.current_mask + 1);
			for (iterator ii = cp.begin(); ii != cp.end(); ii++) insert(*ii);
		}

		unordered_map(unordered_map&& mv)
			: end_iterator(*this, -1, nullptr),
			end_local_iterator(nullptr)
		{
			current_mask = mv.current_mask;
			m_pBuckets = mv.m_pBuckets;

			mv.current_mask = 0;
			mv.m_pBuckets = NULL;
		}

		unordered_map& operator=(const unordered_map& cp)
		{
			clear();
			Reallocate(cp.current_mask + 1);
			for (iterator ii = cp.begin(); ii != cp.end(); ii++) insert(*ii);
			return *this;
		}

		unordered_map& operator=(unordered_map&& mv)			
		{
			current_mask = mv.current_mask;
			m_pBuckets = mv.m_pBuckets;

			mv.current_mask = 0;
			mv.m_pBuckets = NULL;
			return *this;
		}

		void clear()
		{
			if (m_pBuckets != nullptr)
			{
				for (size_type ii = 0; ii < (current_mask+1); ii++)
				{
					node* pNode = m_pBuckets[ii];
					while (pNode != nullptr)
					{
						node* pNextNode = pNode->pNextNode;
						delete pNode;
						pNode = pNextNode;
					}
					m_pBuckets[ii] = nullptr;
				}				
			}
		}

		~unordered_map()
		{
			if (m_pBuckets != nullptr)
			{
				clear();
				delete[] m_pBuckets;
				m_pBuckets = nullptr;
			}
			current_mask = 0;
		}		

		size_type bucket_count() const { return (m_pBuckets != nullptr) ? (current_mask + 1) : 0; }

		iterator begin() const
		{
			for (size_type ii=0; ii < bucket_count(); ii++)			
				if (m_pBuckets[ii] != nullptr) return iterator(*this, ii, m_pBuckets[ii]);
			return end_iterator;
		}

		local_iterator begin(size_type iBucket) const { return local_iterator(m_pBuckets[iBucket]); }

		iterator end() const { return end_iterator; }
		local_iterator end (size_type n) const { return end_local_iterator; }

		iterator find (const key_type& k) const
		{
			hash_type hash = ComputeHash(k);
			size_type iBucket = (hash & current_mask);
			node *pNode = m_pBuckets[iBucket];
			while (pNode != nullptr)
			{
				if (pNode->Value.first == k) 
				{
					iterator ret(*this, iBucket, pNode);
					return ret;
				}
				pNode = pNode->pNextNode;
			}
			return end();
		}

		mapped_type& at (const key_type& k)
		{
			hash_type hash = ComputeHash(k);
			size_type iBucket = (hash & current_mask);
			node *pNode = m_pBuckets[iBucket];
			while (pNode != nullptr)
			{
				if (pNode->Value.first == k) 
				{
					return pNode->Value.second;
				}
				pNode = pNode->pNextNode;
			}
			throw IndexOutOfRangeException();
		}

		pair<iterator,bool> insert ( const value_type& val )
		{
			hash_type hash = ComputeHash(val.first);
			size_type iBucket = (hash & current_mask);			
			node *pNewNode = new node;
			pNewNode->Value = val;
			pNewNode->pNextNode = m_pBuckets[iBucket];
			m_pBuckets[iBucket] = pNewNode;
			return pair<iterator,bool>(iterator(*this, iBucket, m_pBuckets), true);
		}

		template <class P> pair<iterator,bool> insert ( P&& val )
		{
			hash_type hash = ComputeHash(val.first);
			size_type iBucket = (hash & current_mask);
			node *pNewNode = new node;
			pNewNode->Value = std::forward<P>(val);
			pNewNode->pNextNode = m_pBuckets[iBucket];
			m_pBuckets[iBucket] = pNewNode;
			return pair<iterator,bool>(iterator(*this, iBucket, m_pBuckets[iBucket]), true);
		}		

		// Returns the number of elements erased (1 if an element was erased, 0 if it was not found.)
		size_type erase (const key_type& k)
		{
			hash_type hash = ComputeHash(k);
			size_type iBucket = (hash & current_mask);			
			node **ppSrc = &m_pBuckets[iBucket];
			node *pNode = m_pBuckets[iBucket];
			while (pNode != nullptr)
			{
				node* pNextNode = pNode->pNextNode;
				if (pNode->Value.first == k) 
				{					
					delete pNode;
					*ppSrc = pNextNode;
					return 1;
				}
				ppSrc = &(pNode->pNextNode);
				pNode = pNode->pNextNode;
			}
			return 0;			
		}

		size_type size()
		{
			if (m_pBuckets == nullptr) return 0;			
			size_type ret = 0;
			for (size_type ii = 0; ii < (current_mask+1); ii++)
			{
				node* pNode = m_pBuckets[ii];
				while (pNode != nullptr)
				{
					ret ++;
					pNode = pNode->pNextNode;
				}				
			}	
			return ret;
		}

		/** WB Version Extensions to STL Version **/

		void insert(const key_type& key, const mapped_type& value)
		{
			unordered_map<Key,T,Hasher>::insert(typename unordered_map<Key,T,Hasher>::value_type(key,value));
		}

	private:
		void reinsert (node* pNewNode)
		{
			hash_type hash = ComputeHash(pNewNode->Value.first);
			size_type iBucket = (hash & current_mask);
			pNewNode->pNextNode = m_pBuckets[iBucket];
			m_pBuckets[iBucket] = pNewNode;
		}

		iterator end_iterator;
		local_iterator end_local_iterator;
	};	

}

#	endif

#endif	// __WBUnorderedMap_h__

//	End of UnorderedMap.h

