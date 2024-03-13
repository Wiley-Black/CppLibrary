/*	StringHelpers.h
	Copyright (C) 2021 by Wiley Black (TheWiley@gmail.com)	
*/

#ifndef __wbStringHelpers_h__
#define __wbStringHelpers_h__

#define S(psz)		(psz)
#define os(psz)		_T(psz)
typedef signed char schar;

#include <vector>
#include "../Foundation/STL/Text/String.h"
#include "../Platforms/Language.h"

namespace wb
{
	using namespace std;

#if defined(UNICODE) || defined(_MBCS)	
	typedef wstring osstring;
	typedef wchar_t oschar;
#else
	typedef string osstring;
	typedef char oschar;
#endif

	/** Trim Whitespace **/

	// Lacks Unicode/locale support.	
	static inline bool IsWhitespace(char ch)
	{
		return (ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v' || ch == ' ');
	}

	static inline string& TrimStart(string& s) {
		size_t count = 0;
		while (IsWhitespace(s[count])) count++;
		s = s.substr(count);
		return s;
	}

	static inline string& TrimEnd(string& s) {
		size_t index = s.length() - 1;
		while (index < s.length() && IsWhitespace(s[index])) index--;
		s = s.substr(0, index + 1);
		return s;
	}

	static inline string& Trim(string& s) { return TrimStart(TrimEnd(s)); }

	static inline string TrimStart(const string& s) {
		size_t count = 0;
		while (IsWhitespace(s[count])) count++;
		string ret = s.substr(count);
		return ret;
	}

	static inline string TrimEnd(const string& s) {
		size_t index = s.length() - 1;
		while (index < s.length() && IsWhitespace(s[index])) index--;
		string ret = s.substr(0, index + 1);
		return ret;
	}

	static inline string Trim(const string& s) { return TrimStart(TrimEnd(s)); }

	/** to_string() operators, offering quick string conversions for easy display matching and extending C++ STL **/

#if defined(UNICODE) || defined(_MBCS)
	inline std::wstring to_osstring(int value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(long value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(long long value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(unsigned value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(unsigned long value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(unsigned long long value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(float value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(double value) { return std::to_wstring(value); }
	inline std::wstring to_osstring(long double value) { return std::to_wstring(value); }
#else
	inline std::string to_osstring(int value) { return std::to_string(value); }
	inline std::string to_osstring(long value) { return std::to_string(value); }
	inline std::string to_osstring(long long value) { return std::to_string(value); }
	inline std::string to_osstring(unsigned value) { return std::to_string(value); }
	inline std::string to_osstring(unsigned long value) { return std::to_string(value); }
	inline std::string to_osstring(unsigned long long value) { return std::to_string(value); }
	inline std::string to_osstring(float value) { return std::to_string(value); }
	inline std::string to_osstring(double value) { return std::to_string(value); }
	inline std::string to_osstring(long double value) { return std::to_string(value); }
#endif

	/** Additional non-STL to_string() functions **/

	inline string to_string(const char* value) { return string(value); }
#if (defined(_MSC_VER))
	inline string to_hex_string(UInt8 value, bool Uppercase = true) { char ret[5 + 3]; sprintf_s(ret, sizeof(ret), (Uppercase ? "%02X" : "%02x"), value); return ret; }
	inline string to_hex_string(UInt16 value, bool Uppercase = true) { char ret[7 + 3]; sprintf_s(ret, sizeof(ret), (Uppercase ? "%04X" : "%04x"), value); return ret; }
	inline string to_hex_string(unsigned value, bool Uppercase = true) { char ret[8 + 5]; sprintf_s(ret, sizeof(ret), (Uppercase ? "%X" : "%x"), value); return ret; }
	inline string to_hex_string(unsigned long value, bool Uppercase = true) { char ret[8 + 5]; sprintf_s(ret, sizeof(ret), (Uppercase ? "%lX" : "%lx"), value); return ret; }
	inline string to_hex_string(unsigned long long value, bool Uppercase = true) { char ret[16 + 5]; sprintf_s(ret, sizeof(ret), (Uppercase ? "%llX" : "%llx"), value); return ret; }
	inline string to_hex_string(void* pAddr) {
		char ch[20];
#if _WIN64		
		sprintf_s(ch, sizeof(ch), "0x%08llX", (unsigned long long)pAddr);
#else		
		sprintf_s(ch, sizeof(ch), "0x%04X", (unsigned long)pAddr);
#endif
		return string(ch);
	}
	//inline string to_string(byte *pAddr) { return to_string((void *)pAddr); }
#else
	inline string to_hex_string(UInt8 value, bool Uppercase = true) { char ret[5 + 3]; sprintf(ret, (Uppercase ? "%02X" : "%02x"), value); return ret; }
	inline string to_hex_string(UInt16 value, bool Uppercase = true) { char ret[7 + 3]; sprintf(ret, (Uppercase ? "%04X" : "%04x"), value); return ret; }
	inline string to_hex_string(unsigned value, bool Uppercase = true) { char ret[8 + 5]; sprintf(ret, (Uppercase ? "%X" : "%x"), value); return ret; }
	inline string to_hex_string(unsigned long value, bool Uppercase = true) { char ret[8 + 5]; sprintf(ret, (Uppercase ? "%lX" : "%lx"), value); return ret; }
	inline string to_hex_string(unsigned long long value, bool Uppercase = true) { char ret[16 + 5]; sprintf(ret, (Uppercase ? "%llX" : "%llx"), value); return ret; }
#ifdef _X86
	inline string to_hex_string(void* pAddr) { char ch[20]; sprintf(ch, "0x%08X", (unsigned int)pAddr); return string(ch); }
	//inline string to_string(byte *pAddr) { return to_string((void *)pAddr); }
#elif defined(_X64)
	inline string to_hex_string(void* pAddr) { char ch[20]; sprintf(ch, "0x%016llX", (UInt64)pAddr); return string(ch); }
	//inline string to_string(byte *pAddr) { return to_string((void *)pAddr); }
#endif
#endif		

	inline string to_lower(string src) {
		string ret;
		ret.resize(src.length());
		for (size_t ii = 0; ii < src.length(); ii++) ret[ii] = tolower(src[ii]);
		return ret;
	}

	inline string to_lower(const char* psz) {
		string ret;
		ret.resize(strlen(psz));
		for (size_t ii = 0; ii < ret.length(); ii++) ret[ii] = tolower(psz[ii]);
		return ret;
	}

	inline string to_upper(string src) {
		string ret;
		ret.resize(src.length());
		for (size_t ii = 0; ii < src.length(); ii++) ret[ii] = toupper(src[ii]);
		return ret;
	}

	inline string to_upper(const char* psz) {
		string ret;
		ret.resize(strlen(psz));
		for (size_t ii = 0; ii < ret.length(); ii++) ret[ii] = toupper(psz[ii]);
		return ret;
	}

	// Returns 0 if the two strings are identical in a case insensitive comparison.
	inline int compare_no_case(const char* p1, const char* p2)
	{
		register unsigned char* s1 = (unsigned char*)p1;
		register unsigned char* s2 = (unsigned char*)p2;
		unsigned char c1, c2;

		do
		{
			c1 = (unsigned char)toupper((int)*s1++);
			c2 = (unsigned char)toupper((int)*s2++);
			if (c1 == 0) return c1 - c2;
		} 		while (c1 == c2);

		return c1 - c2;
	}

	inline int compare_no_case(const string& lhs, const string& rhs) { return compare_no_case(lhs.c_str(), rhs.c_str()); }
	inline int compare_no_case(const char* lhs, const string& rhs) { return compare_no_case(lhs, rhs.c_str()); }
	inline int compare_no_case(const string& lhs, const char* rhs) { return compare_no_case(lhs.c_str(), rhs); }

	/** Misc **/
	inline bool isnumeric(char c) { return isdigit(c) || c == '+' || c == '-' || c == '.'; }

	/** Replace **/

	inline string Replace(const string& Value, const string& oldSubstr, const string& newSubstr)
	{
		string ret;
		for (size_t ii = 0; ii < Value.size(); )
		{
			bool hit = true;
			for (size_t jj = 0; jj < oldSubstr.size(); jj++)
			{
				if (Value[ii + jj] != oldSubstr[jj])
				{
					hit = false;
					break;
				}
			}
			if (hit)
			{
				ret += newSubstr;
				ii += oldSubstr.size();
			}
			else
			{
				ret += Value[ii];
				ii++;
			}
		}
		return ret;
	}

	/** Simple Tokenizing **/

	enum class StringSplitOptions
	{
		None,
		RemoveEmptyEntries,
		TrimEntries
	};
	AddFlagSupport(StringSplitOptions);

	inline vector<string> Split(const string& source, char separator, StringSplitOptions options = StringSplitOptions::None)
	{
		// Example: ",ONE,,TWO,,,THREE,,"
		// Delimiter character is ','.
		// With no options:				<><ONE><><TWO><><><THREE><><>
		// With RemoveEmptyEntries:		<ONE><TWO><THREE>

		bool RemoveEmpty = (options & StringSplitOptions::RemoveEmptyEntries) != 0;
		bool TrimEntries = (options & StringSplitOptions::TrimEntries) != 0;

		vector<string> ret;
		size_t ii = 0;
		string current;
		while (ii < source.length())
		{
			if (source[ii] == separator)
			{
				if (TrimEntries) Trim(current);
				if (!RemoveEmpty || !current.empty()) ret.push_back(current);
				current.clear();
				ii++;
			}
			else
			{
				current += source[ii];
				ii++;
			}
		}
		if (TrimEntries) Trim(current);
		if (!RemoveEmpty || !current.empty()) ret.push_back(current);
		return ret;
	}
}

#endif	// __wbStringHelpers_h__

//	End of StringHelpers.h



