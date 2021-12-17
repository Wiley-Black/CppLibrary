/*	Encoding.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBEncoding_h__
#define __WBEncoding_h__

/** Table of contents / References **/

namespace wb {
	namespace text {
		class Encoding;
		class UTF8Encoding;
	};
};

/** Dependencies **/

#include "../Platforms/Platforms.h"
#include "../Foundation/STL/Text/String.h"
#include "StringHelpers.h"

//#include <cctype>
//#include <cwctype>
//#include <string>	

/** Content **/

namespace wb
{
	namespace text
	{

		/** Encodings **/

		class Encoding
		{
		public:
			virtual uint GetCodePage() const = 0;

			static UTF8Encoding GetUTF8();
			#if _MSC_VER >= 1900			// Requires Visual Studio 2015 for "magic statics", a C++11 feature.
			static const Encoding& GetDefault();
			#endif
		};

		class UTF8Encoding : public Encoding
		{
		public:
			uint GetCodePage() const override { return 65001; }
		};

		/** Statics **/

		inline /*static*/ UTF8Encoding Encoding::GetUTF8() { return UTF8Encoding(); }

		#if _MSC_VER >= 1900			// Requires Visual Studio 2015 for "magic statics", a C++11 feature.
		inline /*static*/ const Encoding& Encoding::GetDefault()
		{
			static UTF8Encoding Default;
			return *(Encoding*)&Default;
		}
		// An alternative implementation would rely on the PrimaryModule mechanism to include a static copy in
		// a single compilation unit.
		#endif
	}

	/** Conversions **/

	inline string to_string(const string& str, const text::Encoding& ToEncoding = text::Encoding::GetDefault()) { return str; }
	inline wstring to_wstring(const wstring& str, const text::Encoding& ToEncoding = text::Encoding::GetDefault()) { return str; }

}// End namespace wb

/** Late Dependencies **/

#include "../Foundation/Exceptions.h"

namespace wb
{
	/// <summary>to_string() accepts a UTF-16 input string 'str' and converts it to a std::string in the specified encoding.</summary>
	inline string to_string(const wstring& str, const text::Encoding& ToEncoding = text::Encoding::GetDefault())
	{
		size_t sz_full = str.length();
		if (sz_full > Int32_MaxValue) throw ArgumentOutOfRangeException("Exceeded maximum supported string length for this conversion.");
		int sz = (int)sz_full;
#if defined(_WINDOWS)
		int nd = WideCharToMultiByte(ToEncoding.GetCodePage(), 0, &str[0], sz, NULL, 0, NULL, NULL);
		string ret(nd, 0);
		int w = WideCharToMultiByte(ToEncoding.GetCodePage(), 0, &str[0], sz, &ret[0], nd, NULL, NULL);
		if (w != nd) {
			throw Exception(S("Invalid size written during wide string to multibyte string conversion."));
		}
		return ret;
#else
		const wchar_t* p = str.c_str();
		char* tp = new char[sz];
		size_t w = wcstombs(tp, p, sz);
		if (w != (size_t)sz) {
			delete[] tp;
			throw Exception(S("Invalid size written during wide string to multibyte string conversion."));
		}
		string ret(tp);
		delete[] tp;
		return ret;
#endif
	}

	/// <summary>to_wstring() accepts an input string 'str' given in the specified encoding and converts it to a UTF-16 std::wstring.</summary>
	inline wstring to_wstring(const string& str, const text::Encoding& FromEncoding = text::Encoding::GetDefault())
	{
#if defined(_WINDOWS)
		size_t sz_full = str.length();
		if (sz_full > Int32_MaxValue) throw ArgumentOutOfRangeException("Exceeded maximum supported string length for this conversion.");
		int sz = (int)sz_full;
		int nd = MultiByteToWideChar(FromEncoding.GetCodePage(), 0, &str[0], sz, NULL, 0);
		wstring ret(nd, 0);
		int w = MultiByteToWideChar(FromEncoding.GetCodePage(), 0, &str[0], sz, &ret[0], nd);
		if (w != nd) {
			throw Exception(S("Invalid size written during wide string to multibyte string conversion."));
		}
		return ret;
#else
		const char* p = str.c_str();
		size_t len = str.length();
		size_t sz = len * sizeof(wchar);
		wchar* tp = new wchar[sz];
		size_t w = mbstowcs(tp, p, sz);
		if (w != len) {
			delete[] tp;
			throw Exception(S("Invalid size written during wide string to multibyte string conversion."));
		}
		wstring ret(tp);
		delete[] tp;
		return ret;
#endif
	}

#if defined(UNICODE) || defined(_MBCS)
	inline wstring to_osstring(const string& str, const text::Encoding& FromEncoding = text::Encoding::GetDefault()) { return to_wstring(str); }
	inline wstring to_osstring(const wstring& str, const text::Encoding& FromEncoding = text::Encoding::GetDefault()) { return str; }
#else
	inline string to_osstring(const string& str, const text::Encoding& FromEncoding = text::Encoding::GetDefault()) { return str; }
	inline string to_osstring(const wstring& str, const text::Encoding& FromEncoding = text::Encoding::GetDefault()) { return to_string(str); }
#endif
}

#endif	// __WBString_h__

//	End of String.h


