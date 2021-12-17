/////////
//  StringComparison.h
//  Copyright (C) 2013-2014 by Wiley Black
//	Author(s):
//		Wiley Black			TheWiley@gmail.com
////

#ifndef __WBStringComparison_h__ 
#define __WBStringComparison_h__

/** Dependencies **/

#include "../Platforms/Platforms.h"

/** Content **/

namespace wb
{
	// Case-Sensitive comparisons

	inline bool IsEqual(const char *pszA, const char *pszB)
	{
		for (;;)
		{
			if (*pszA != *pszB) return false;
			if (*pszA == 0) return true;		// Implies that *pszB also = 0 from above check.
			pszA ++; pszB ++;
		}
	}

	inline bool IsEqual(const string& strA, const string& strB)
	{	
		if (strA.length() != strB.length()) return false;
		for (size_t ii=0; ii < strA.length(); ii++)
		{
			if (strA[ii] != strB[ii]) return false;
		}
		return true;
	}

	inline bool IsEqual(const string& strA, const char *pszB)
	{		
		for (size_t ii=0; ii < strA.length(); ii++, pszB++)
		{		
			if (strA[ii] != *pszB) return false;
			if (*pszB == 0) return false;
		}
		return (*pszB == 0);
	}

	inline bool IsEqual(const char *pszA, const string& strB) { return IsEqual(strB, pszA); }

	inline bool IsEqual(const wstring& strA, const wstring& strB)
	{	
		if (strA.length() != strB.length()) return false;
		for (size_t ii=0; ii < strA.length(); ii++)
		{
			if (strA[ii] != strB[ii]) return false;
		}
		return true;
	}

		// i.e. StartsWith("A full string", "A full") will return true.
	inline bool StartsWith(const char *psz, const char *pszSubstring) 
	{ 
		for (;;)
		{
			if (*pszSubstring == 0) return true;
			if (*psz == 0) return false;
			if (*psz != *pszSubstring) return false;
			psz ++; pszSubstring ++;
		}
	}

	inline bool StartsWith(const string& str, const char *pszSubstring) { return StartsWith(str.c_str(), pszSubstring); }
	inline bool StartsWith(const string& str, const string& strSubstring) { return StartsWith(str.c_str(), strSubstring.c_str()); }
	inline bool StartsWith(const char* psz, const string& strSubstring) { return StartsWith(psz, strSubstring.c_str()); }

		// i.e. StartsWith("A full string", "A FULL") will return true.
	inline bool StartsWithNoCase(const char *psz, const char *pszSubstring) 
	{ 
		for (;;)
		{
			if (*pszSubstring == 0) return true;
			if (*psz == 0) return false;
			if (tolower(*psz) != tolower(*pszSubstring)) return false;
			psz ++; pszSubstring ++;
		}
	}

	inline bool StartsWithNoCase(const string& str, const char *pszSubstring) { return StartsWithNoCase(str.c_str(), pszSubstring); }
	inline bool StartsWithNoCase(const string& str, const string& strSubstring) { return StartsWithNoCase(str.c_str(), strSubstring.c_str()); }
	inline bool StartsWithNoCase(const char* psz, const string& strSubstring) { return StartsWithNoCase(psz, strSubstring.c_str()); }

	inline int IndexOf(const char *psz, const char *pszSubstring)
	{
		if (*pszSubstring == 0) return -1;
		for (int index = 0; ; index++, psz++)
		{
			if (*psz == 0) return -1;
			if (*psz == *pszSubstring)
			{
				const char *p1 = psz + 1;
				const char *ps = pszSubstring + 1;
				for (;;)
				{				
					if (*ps == 0) return index;
					if (*p1 != *ps) break;
					p1 ++;
					ps ++;
				}
			}
		}
	}

	// Case-Insensitive comparisons

	inline bool IsEqualNoCase(const char *pszA, const char *pszB)
	{
		for (;;)
		{
			if (tolower(*pszA) != tolower(*pszB)) return false;
			if (*pszA == 0) return true;		// Implies that *pszB also = 0 from above check.
			pszA ++; pszB ++;
		}
	}

	inline bool IsEqualNoCase(const string& strA, const string& strB)
	{	
		if (strA.length() != strB.length()) return false;
		for (size_t ii=0; ii < strA.length(); ii++)
		{
			if (tolower(strA[ii]) != tolower(strB[ii])) return false;
		}
		return true;
	}

	inline bool IsEqualNoCase(const string& strA, const char *pszB)
	{		
		for (size_t ii=0; ii < strA.length(); ii++, pszB++)
		{		
			if (tolower(strA[ii]) != tolower(*pszB)) return false;
			if (*pszB == 0) return false;
		}
		return (*pszB == 0);
	}

	inline bool IsEqualNoCase(const char *pszA, const string& strB) { return IsEqualNoCase(strB, pszA); }

	inline bool IsEqualNoCase(const wstring& strA, const wstring& strB)
	{	
		if (strA.length() != strB.length()) return false;
		for (size_t ii=0; ii < strA.length(); ii++)
		{
			if (tolower(strA[ii]) != tolower(strB[ii])) return false;
		}
		return true;
	}

	// Other Misc. string functions

	inline int CountWhitespace(const char *psz)
	{
		int Count = 0;
		while (*psz == ' ' || *psz == '\t' || *psz == '\r' || *psz == '\n') { Count ++; psz++; }
		return Count;
	}
}

#endif	// __StringComparison_h__

//	End of StringComparison.h

