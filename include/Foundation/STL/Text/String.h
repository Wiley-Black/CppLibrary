/*	String.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)

	The wb::string type uses a single-byte character encoding method.  Because the DML library
	uses UTF-8 strings, the wb::string type matches this.  This also eliminates the need for
	a _T() macro around strings.  However, environments such as Windows often expect a wide-byte
	string (wstring) for OS operations.  Since the underlying wb::string type can store UTF-8
	strings supporting all languages (albiet the character vs element count is inaccurate), the
	wb::string type is chosen as the default and conversion to wstring is provided via to_wstring.

	When UNICODE is defined, a type called osstring is mapped to wstring.  When UNICODE is not
	defined, osstring maps to string.

	When the STL library is used, the wb::string and wb::wstring classes are not declared.

	For later modification of this behavior, an optional macro S() is provided that is similar to the _T() macro
	but provides a version of the string literal compatible with wb::string.  A macro os() is provided that
	provides a version of the string literal compatible with wb::osstring.  A macro similar to TCHAR called SCHAR
	is also available which presently maps to char.
*/

#ifndef __WBString_h__
#define __WBString_h__

#define S(psz)		(psz)
#define os(psz)		_T(psz)
typedef signed char schar;

#include <string.h>
#include "../StdException.h"

#ifndef EmulateSTL
#include <string>
#include <algorithm>				// For find_if, used by the Trim..() functions.
#include <functional>				// For not1, used by the Trim..() functions.
#include <locale>					// For isspace, used by the Trim..() functions.
#else

#include <stdio.h>
#include <ctype.h>
#if (defined(_MSC_VER))
#include <tchar.h>
#endif

namespace std
{
	/** basic_string template class **/

	template<class Elem> class basic_string
	{
		/// <summary>m_pData contains the underlying string.  The m_pData buffer is always allocated to accomodate
		/// at least one character beyond m_nLength so that a null terminator can be set, however the null-terminating
		/// character is not written until c_str() is called.<summary>
		Elem* m_pData;

		size_t m_nLength;		// In units of elements
		size_t m_nAllocated;	// In units of elements

		void Reallocate(size_t nCurrentNeed)
		{
			nCurrentNeed ++;		// Add one to ensure room for a null terminator when needed.
			if (nCurrentNeed <= m_nAllocated) return;
			if (nCurrentNeed < 64) m_nAllocated = 64;
			else if (nCurrentNeed < 256) m_nAllocated = 256;
			else if (nCurrentNeed < 2048) m_nAllocated = 2048;
			else if (nCurrentNeed < 65536) m_nAllocated = 65536;
			else m_nAllocated = ((nCurrentNeed - 1) & ~0xFFFF) + 0x10000;		// Round up to nearest 64K block size.

			Elem* pNewData = new Elem [m_nAllocated];
			memcpy(pNewData, m_pData, m_nLength * sizeof(Elem));
			if (m_pData != nullptr) delete[] m_pData;
			m_pData = pNewData;
		}		

		template<class Elem> friend basic_string<Elem> operator+(const basic_string<Elem>& lhs, const basic_string<Elem>& rhs);
		template<class Elem> friend basic_string<Elem> operator+(basic_string<Elem>&&      lhs, basic_string<Elem>&&      rhs);
		template<class Elem> friend basic_string<Elem> operator+(basic_string<Elem>&&      lhs, const basic_string<Elem>& rhs);
		template<class Elem> friend basic_string<Elem> operator+(const basic_string<Elem>& lhs, Elem          rhs);
		template<class Elem> friend basic_string<Elem> operator+(const basic_string<Elem>& lhs, const Elem*   rhs);
		template<class Elem> friend basic_string<Elem> operator+(const Elem*   lhs, const basic_string<Elem>& rhs);
		template<class Elem> friend basic_string<Elem> operator+(Elem          lhs, const basic_string<Elem>& rhs);
		template<class Elem> friend basic_string<Elem> operator+(basic_string<Elem>&&      lhs, const Elem*   rhs);
		template<class Elem> friend basic_string<Elem> operator+(basic_string<Elem>&&      lhs, Elem          rhs);

		/** STL does not gaurantee contiguous underlying memory, so avoid accessing
			the string memory directly unless implementation-specific code is known. **/		
		void reserve(size_t n = 0) { Reallocate(n); }

	public:

		static size_t szlen(const Elem* psz)
		{
			Elem *p = (Elem*)psz;
			size_t ret = 0;
			while (*p != 0) { ret++; p++; }
			return ret;
		}	

		static const size_t npos = -1;

		/** Constructors **/

		basic_string() 
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{ }
		basic_string(const basic_string<Elem>& str) 
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{
			Reallocate(str.m_nLength);
			memcpy(m_pData, str.m_pData, str.m_nLength * sizeof(Elem));
			m_nLength = str.m_nLength;
		}
		basic_string(const basic_string<Elem>& str, size_t pos, size_t len = npos)
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{
			if (len == npos || pos + len > str.m_nLength) len = str.m_nLength - pos;
			Reallocate(len);
			memcpy(m_pData, str.m_pData + pos, len * sizeof(Elem));
			m_nLength = len;
		}
		basic_string(const Elem* s)
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{
			size_t nLength = szlen(s);
			Reallocate(nLength);
			memcpy(m_pData, s, nLength * sizeof(Elem));
			m_nLength = nLength;
		}
		basic_string(const Elem* s, size_t n)
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{
			Reallocate(n);
			memcpy(m_pData, s, n * sizeof(Elem));
			m_nLength = n;
		}
		basic_string(size_t n, Elem c)
			: m_pData(NULL), m_nLength(0), m_nAllocated(0)
		{			
			Reallocate(n);
			for (size_t ii=0; ii < n; ii++) m_pData[ii] = c;
			m_nLength = n;
		}
		// template <class InputIterator> sstring  (InputIterator first, InputIterator last);
		// sstring (initializer_list<char> il);
		basic_string(basic_string<Elem>&& str)
		{
			m_pData = str.m_pData;
			m_nAllocated = str.m_nAllocated;
			m_nLength = str.m_nLength;
			str.m_pData = nullptr;
			str.m_nAllocated = 0;
		}

		/** Cleanup **/
		~basic_string()
		{
			if (m_pData != nullptr) delete[] m_pData;
			m_pData = nullptr;
			m_nAllocated = 0;
			m_nLength = 0;
		}

		/** Assignment **/

		basic_string<Elem>& operator=(const basic_string<Elem>& str)
		{
			m_nLength = 0;
			Reallocate(str.m_nLength);
			memcpy(m_pData, str.m_pData, str.m_nLength * sizeof(Elem));
			m_nLength = str.m_nLength;
			return *this;
		}

		basic_string<Elem>& operator=(const Elem* s)
		{
			m_nLength = 0;
			size_t nLength = szlen(s);
			Reallocate(nLength);
			memcpy(m_pData, s, nLength * sizeof(Elem));
			m_nLength = nLength;
			return *this;
		}

		basic_string<Elem>& operator=(Elem c)
		{
			m_nLength = 0;
			Reallocate(1);
			m_nLength = 1;
			m_pData[0] = c;
			return *this;
		}

		// sstring& operator= (initializer_list<char> il);

		basic_string<Elem>& operator=(basic_string<Elem>&& str)
		{
			if (m_pData != nullptr) delete[] m_pData;
			m_pData = str.m_pData; str.m_pData = nullptr;
			m_nAllocated = str.m_nAllocated; str.m_nAllocated = 0;
			m_nLength = str.m_nLength; str.m_nLength = 0;
			return *this;
		}

		/** Accessors **/		

		const Elem* c_str() const
		{
			m_pData[m_nLength] = 0;				// Set null terminator.
			return m_pData;
		}

		size_t length() const { return m_nLength; }
		size_t size() const { return m_nLength * sizeof(Elem); }

		Elem& operator[] (size_t pos) { return m_pData[pos]; }
		const Elem& operator[] (size_t pos) const { return m_pData[pos]; }

		/** Operations **/		

		basic_string<Elem>& operator+=(const basic_string<Elem>& str)
		{
			Reallocate(m_nLength + str.m_nLength);
			memcpy(m_pData + m_nLength, str.m_pData, str.m_nLength * sizeof(Elem));
			m_nLength += str.m_nLength;
			return *this;
		}
		
		basic_string<Elem>& operator+=(const Elem* s)
		{
			size_t nAdd = szlen(s);
			Reallocate(m_nLength + nAdd);
			memcpy(m_pData + m_nLength, s, nAdd * sizeof(Elem));
			m_nLength += nAdd;
			return *this;
		}
		
		basic_string<Elem>& operator+=(Elem c)
		{
			Reallocate(m_nLength + 1);
			m_pData[m_nLength++] = c;
			return *this;
		}

		// sstring& operator+= (initializer_list<char> il);

		basic_string<Elem>& append (const basic_string<Elem>& str) { return operator+=(str); }		
		basic_string<Elem>& append (const Elem* s) { return operator+=(s); }

		basic_string<Elem>& append (const basic_string<Elem>& str, size_t subpos, size_t sublen)
		{
			if (sublen == npos) sublen = str.length() - subpos;
			if (sublen > str.length() - subpos) throw exception("Substring length exceeds source string length.");
			Reallocate(m_nLength + sublen);
			memcpy(m_pData + m_nLength, str.m_pData + subpos, sublen * sizeof(Elem));
			m_nLength += sublen;
			return *this;
		}

		basic_string<Elem>& append (const Elem* s, size_t n)
		{
			size_t nAdd = szlen(s);
			if (n > nAdd) throw exception("Substring length exceeds source string length.");			
			nAdd = n;
			Reallocate(m_nLength + nAdd);
			memcpy(m_pData + m_nLength, s, nAdd * sizeof(Elem));
			m_nLength += nAdd;
			return *this;
		}

		basic_string<Elem>& append (size_t n, Elem c)
		{
			Reallocate(m_nLength + n);
			for (int ii=0; ii < n; ii++) m_pData[m_nLength++] = c;
			return *this;
		}		

		// range (6)	template <class InputIterator> string& append (InputIterator first, InputIterator last);

		void clear() { m_nLength = 0; }

		basic_string<Elem> substr (size_t pos = 0, size_t len = npos) const
		{
			return basic_string<Elem>(*this, pos, len);
		}

		void resize (size_t n)
		{
			Reallocate(n);
			if (n > m_nLength) 
				memset(&m_pData[m_nLength], 0, (n - m_nLength) * sizeof(Elem));
			m_nLength = n;
		}

		void resize (size_t n, Elem c)
		{
			Reallocate(n);
			if (n > m_nLength) 
				for (size_t ii = m_nLength; ii < n; ii++) m_pData[ii] = c;
			m_nLength = n;
		}

		/** Comparisons **/

		int compare (const basic_string<Elem>& str) const  
		{ 
			if (str.length() < m_nLength) return -1;
			if (str.length() > m_nLength) return 1;

			for (size_t ii=0; ii < m_nLength; ii++)
				if (str[ii] != m_pData[ii])
					return (str[ii] < m_pData[ii]) ? -1 : 1;
			return 0;
		}

		int compare (const Elem* s) const 
		{ 
			size_t nStrLength = szlen(s);
			if (nStrLength < m_nLength) return -1;
			if (nStrLength > m_nLength) return 1;

			for (size_t ii=0; ii < m_nLength; ii++)
				if (s[ii] != m_pData[ii])
					return (s[ii] < m_pData[ii]) ? -1 : 1;
			return 0;
		}

		/** Searches **/		

		size_t find_first_of (const basic_string<Elem>& str, size_t pos = 0) const { return find_first_of(str.m_pData, pos, str.m_nLength); }

		size_t find_first_of (const Elem* s, size_t pos = 0) const { return find_first_of(s, pos, strlen(s)); }

		size_t find_first_of (const Elem* s, size_t pos, size_t n) const
		{
			while (pos < m_nLength)
			{
				for (size_t ii=0; ii < n; ii++)
					if (m_pData[pos] == s[ii]) return pos;
				pos++;
			}
			return npos;
		}

		size_t find_first_of (Elem c, size_t pos = 0) const 
		{
			while (pos < m_nLength)
			{
				if (m_pData[pos] == c) return pos;
				pos++;
			}
			return npos;
		}
		
		size_t find_last_of (const basic_string<Elem>& str, size_t pos = npos) const { return find_last_of(str.m_pData, pos, str.m_nLength); }
		size_t find_last_of (const Elem* s, size_t pos = npos) const { return find_last_of(s, pos, strlen(s)); }

		size_t find_last_of (const Elem* s, size_t pos, size_t n) const
		{
			if (pos >= m_nLength) pos = m_nLength - 1;
			while (pos < m_nLength)
			{
				for (size_t ii=0; ii < n; ii++)				
					if (m_pData[pos] == s[ii]) return pos;
				
				pos--;
			}
			return npos;
		}

		size_t find_last_of (Elem c, size_t pos = npos) const 
		{
			if (pos >= m_nLength) pos = m_nLength - 1;
			while (pos < m_nLength)
			{
				if (m_pData[pos] == c) return pos;
				pos--;
			}
			return npos;
		}
		
		size_t find (const basic_string<Elem>& str, size_t pos = 0) const { return find(str.m_pData, pos, str.m_nLength); }		
		size_t find (const Elem* s, size_t pos = 0) const { return find(s, pos, strlen(s)); }
		
		size_t find (const Elem* s, size_t pos, size_t n) const
		{
			while (pos < m_nLength)
			{
				int ii = 0;
				for (;;)
				{
					if (m_pData[pos+ii] != s[ii]) break;
					ii++;
					if (ii == n) return pos;
				}
				pos ++;
			}
			return npos;
		}
		
		size_t find (Elem c, size_t pos = 0) const { return find_first_of(c, pos); }

		size_t rfind (const basic_string<Elem>& str, size_t pos = npos) const { return find(str.m_pData, pos, str.m_nLength); }		
		size_t rfind (const Elem* s, size_t pos = npos) const { return find(s, pos, strlen(s)); }
		
		size_t rfind (const Elem* s, size_t pos, size_t n) const
		{
			if (pos >= m_nLength) pos = m_nLength - 1;
			while (pos < m_nLength)
			{
				int ii = 0;
				for (;;)
				{
					if (m_pData[pos+ii] != s[ii]) break;
					ii++;
					if (ii == n) return pos;
				}
				pos --;
			}
			return npos;
		}
		
		size_t rfind (Elem c, size_t pos = npos) const { return find_last_of(c, pos); }
	};

	/** basic_string<Elem> operator+ implementations (full copies) **/
	 
	template<class Elem> inline basic_string<Elem> operator+ (const basic_string<Elem>& lhs, const basic_string<Elem>& rhs)
	{
		basic_string<Elem> ret;
		size_t nLength = lhs.m_nLength + rhs.m_nLength;
		ret.reserve(nLength);
		memcpy(ret.m_pData, lhs.m_pData, lhs.m_nLength * sizeof(Elem));
		memcpy(ret.m_pData + lhs.m_nLength, rhs.m_pData, rhs.m_nLength * sizeof(Elem));		
		ret.m_nLength = nLength;
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (const basic_string<Elem>& lhs, basic_string<Elem>&&      rhs)
	{
		return operator+ (lhs, (const basic_string<Elem>&)rhs);
	}	

	template<class Elem> inline basic_string<Elem> operator+ (const basic_string<Elem>& lhs, const Elem*   rhs)
	{
		size_t nRhsLength = basic_string<Elem>::szlen(rhs);
		basic_string<Elem> ret;
		size_t nLength = lhs.m_nLength + nRhsLength;
		ret.reserve(nLength);		
		memcpy(ret.m_pData, lhs.m_pData, lhs.m_nLength * sizeof(Elem));
		memcpy(ret.m_pData + lhs.m_nLength, rhs, nRhsLength * sizeof(Elem));
		ret.m_nLength = nLength;
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (const Elem*   lhs, const basic_string<Elem>& rhs)
	{
		size_t nLhsLength = basic_string<Elem>::szlen(lhs);
		basic_string<Elem> ret;
		size_t nLength = nLhsLength + rhs.m_nLength;
		ret.reserve(nLength);
		memcpy(ret.m_pData, lhs, nLhsLength * sizeof(Elem));
		memcpy(ret.m_pData + nLhsLength, rhs.m_pData, rhs.m_nLength * sizeof(Elem));
		ret.m_nLength = nLength;
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (const Elem*   lhs, basic_string<Elem>&&      rhs)
	{
		return operator+ (lhs, (const basic_string<Elem>&)rhs);
	}

	template<class Elem> inline basic_string<Elem> operator+ (const basic_string<Elem>& lhs, Elem          rhs)
	{
		basic_string<Elem> ret;
		size_t nLength = lhs.m_nLength + 1;
		ret.reserve(nLength);
		memcpy(ret.m_pData, lhs.m_pData, lhs.m_nLength * sizeof(Elem));
		ret.m_pData[lhs.m_nLength] = rhs;
		ret.m_nLength = nLength;
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (Elem          lhs, const basic_string<Elem>& rhs)
	{
		basic_string<Elem> ret;
		size_t nLength = 1 + rhs.m_nLength;
		ret.reserve(nLength);
		ret.m_pData[0] = lhs;
		memcpy(ret.m_pData + 1, rhs.m_pData, rhs.m_nLength * sizeof(Elem));
		ret.m_nLength = nLength;
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (Elem          lhs, basic_string<Elem>&&      rhs)
	{
		return operator+ (lhs, (const basic_string<Elem>&)rhs);
	}

	/** sstring operator+ implementations (move-copies) **/

	template<class Elem> inline basic_string<Elem> operator+ (basic_string<Elem>&&      lhs, const basic_string<Elem>& rhs)
	{
		basic_string<Elem> ret;
		ret.m_pData = lhs.m_pData; 
		ret.m_nAllocated = lhs.m_nAllocated; 
		ret.m_nLength = lhs.m_nLength; 
		size_t nLength = lhs.m_nLength + rhs.m_nLength;
		ret.reserve(nLength);
		memcpy(ret.m_pData + lhs.m_nLength, rhs.m_pData, rhs.m_nLength * sizeof(Elem));
		ret.m_nLength = nLength;
		lhs.m_pData = nullptr; lhs.m_nAllocated = 0; lhs.m_nLength = 0;		
		return ret;
	}

	template<class Elem> inline basic_string<Elem> operator+ (basic_string<Elem>&&      lhs, basic_string<Elem>&&      rhs)
	{
		return operator+ (lhs, (const basic_string<Elem>&)rhs);
	}	

	template<class Elem> inline basic_string<Elem> operator+ (basic_string<Elem>&&      lhs, const Elem*   rhs)
	{
		size_t nRhsLength = basic_string<Elem>::szlen(rhs);
		basic_string<Elem> ret;
		ret.m_pData = lhs.m_pData; 
		ret.m_nAllocated = lhs.m_nAllocated; 
		size_t nLength = lhs.m_nLength + nRhsLength;
		ret.reserve(nLength);
		memcpy(ret.m_pData + lhs.m_nLength, rhs, nRhsLength * sizeof(Elem));		
		ret.m_nLength = nLength;
		lhs.m_pData = nullptr; lhs.m_nAllocated = 0; lhs.m_nLength = 0;
		return ret;
	}		

	template<class Elem> inline basic_string<Elem> operator+ (basic_string<Elem>&&      lhs, Elem          rhs)
	{		
		basic_string<Elem> ret;
		ret.m_pData = lhs.m_pData; 
		ret.m_nAllocated = lhs.m_nAllocated; 
		size_t nLength = lhs.m_nLength + 1;
		ret.reserve(nLength);		
		ret.m_pData[lhs.m_nLength] = rhs;
		ret.m_nLength = nLength;
		lhs.m_pData = nullptr; lhs.m_nAllocated = 0;  lhs.m_nLength = 0;
		return ret;
	}	

	typedef basic_string<char>		string;
	typedef basic_string<wchar_t>	wstring;

	#ifdef _MSC_VER	
	inline string to_string(int value) { char ret[11 + 5]; sprintf_s(ret, sizeof(ret), "%d", value); return ret; }
	inline string to_string(long value) { char ret[11 + 5]; sprintf_s(ret, sizeof(ret), "%ld", value); return ret; }
	inline string to_string(long long value) { char ret[21 + 5]; sprintf_s(ret, sizeof(ret), "%lld", value); return ret; }
	inline string to_string(unsigned value) { char ret[11 + 5]; sprintf_s(ret, sizeof(ret), "%u", value); return ret; }
	inline string to_string(unsigned long value) { char ret[11 + 5]; sprintf_s(ret, sizeof(ret), "%lu", value); return ret; }
	inline string to_string(unsigned long long value) { char ret[21 + 5]; sprintf_s(ret, sizeof(ret), "%llu", value); return ret; }
	inline string to_string(float value) { char ret[20]; sprintf_s(ret, sizeof(ret), "%f", value); return ret; }
	inline string to_string(double value) { char ret[40]; sprintf_s(ret, sizeof(ret), "%f", value); return ret; }
	inline string to_string(long double value) { char ret[100]; sprintf_s(ret, sizeof(ret), "%Lf", value); return ret; }
	#else
	inline string to_string(int value) { char ret[11 + 5]; sprintf(ret, "%d", value); return ret; }
	inline string to_string(long value) { char ret[11 + 5]; sprintf(ret, "%ld", value); return ret; }
	inline string to_string(long long value) { char ret[21 + 5]; sprintf(ret, "%lld", value); return ret; }
	inline string to_string(unsigned value) { char ret[11 + 5]; sprintf(ret, "%u", value); return ret; }
	inline string to_string(unsigned long value) { char ret[11 + 5]; sprintf(ret, "%lu", value); return ret; }
	inline string to_string(unsigned long long value) { char ret[21 + 5]; sprintf(ret, "%llu", value); return ret; }
	inline string to_string(float value) { char ret[20]; sprintf(ret, "%f", value); return ret; }
	inline string to_string(double value) { char ret[40]; sprintf(ret, "%f", value); return ret; }
	inline string to_string(long double value) { char ret[100]; sprintf(ret, "%Lf", value); return ret; }
	#endif

#ifdef _MSC_VER	
	inline wstring to_wstring(int value) { wchar_t ret[11 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%d", value); return ret; }
	inline wstring to_wstring(long value) { wchar_t ret[11 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%ld", value); return ret; }
	inline wstring to_wstring(long long value) { wchar_t ret[21 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%lld", value); return ret; }
	inline wstring to_wstring(unsigned value) { wchar_t ret[11 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%u", value); return ret; }
	inline wstring to_wstring(unsigned long value) { wchar_t ret[11 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%lu", value); return ret; }
	inline wstring to_wstring(unsigned long long value) { wchar_t ret[21 + 5]; swprintf_s(ret, sizeof(ret)/2, L"%llu", value); return ret; }
	inline wstring to_wstring(float value) { wchar_t ret[20]; swprintf_s(ret, sizeof(ret)/2, L"%f", value); return ret; }
	inline wstring to_wstring(double value) { wchar_t ret[40]; swprintf_s(ret, sizeof(ret)/2, L"%f", value); return ret; }
	inline wstring to_wstring(long double value) { wchar_t ret[100]; swprintf_s(ret, sizeof(ret)/2, L"%Lf", value); return ret; }
#else
	inline wstring to_wstring(int value) { wchar_t ret[11 + 5]; swprintf(ret, L"%d", value); return ret; }
	inline wstring to_wstring(long value) { wchar_t ret[11 + 5]; swprintf(ret, L"%ld", value); return ret; }
	inline wstring to_wstring(long long value) { wchar_t ret[21 + 5]; swprintf(ret, L"%lld", value); return ret; }
	inline wstring to_wstring(unsigned value) { wchar_t ret[11 + 5]; swprintf(ret, L"%u", value); return ret; }
	inline wstring to_wstring(unsigned long value) { wchar_t ret[11 + 5]; swprintf(ret, L"%lu", value); return ret; }
	inline wstring to_wstring(unsigned long long value) { wchar_t ret[21 + 5]; sprintf(ret, L"%llu", value); return ret; }
	inline wstring to_wstring(float value) { wchar_t ret[20]; swprintf(ret, L"%f", value); return ret; }
	inline wstring to_wstring(double value) { wchar_t ret[40]; swprintf(ret, L"%f", value); return ret; }
	inline wstring to_wstring(long double value) { wchar_t ret[100]; swprintf(ret, L"%Lf", value); return ret; }
#endif
}

#endif	// EmulateSTL

#endif	// __WBString_h__

//	End of String.h
