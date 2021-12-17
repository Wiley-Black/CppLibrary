/*	BaseTypeParsing.h
	Copyright (C) 2014-2016 by Wiley Black (TheWiley@gmail.com)
*/


#ifndef __WBBaseTypeParsing_h__
#define __WBBaseTypeParsing_h__

/** Dependencies **/

#include "../Platforms/Platforms.h"
#include "../Platforms/Language.h"

/** Content **/

namespace wb
{		
	enum_class_start(NumberStyles,int)
	{
		AllowBasePrefix		= 0x01,		// If present, allow prefixes to select base (0x or 0X for base-16 or b for base-2).

		AllowThousands		= 0x10,
		AllowLeadingWhite	= 0x20,
		AllowLeadingSign	= 0x40,
		AllowHexSpecifier	= 0x80,

		Integer				= AllowLeadingWhite | AllowLeadingSign | AllowBasePrefix,
		Float				= AllowLeadingWhite | AllowLeadingSign,
		HexNumber			= AllowLeadingWhite | AllowHexSpecifier,
	}
	enum_class_end(NumberStyles);

	/**
	inline NumberStyles	operator&(NumberStyles x, NumberStyles y) { return static_cast<NumberStyles>(static_cast<int>(x) & static_cast<int>(y)); }
	inline NumberStyles	operator|(NumberStyles x, NumberStyles y) { return static_cast<NumberStyles>(static_cast<int>(x) | static_cast<int>(y)); }
	inline NumberStyles	operator^(NumberStyles x, NumberStyles y) { return static_cast<NumberStyles>(static_cast<int>(x) ^ static_cast<int>(y)); }
	inline NumberStyles	operator~(NumberStyles x) { return static_cast<NumberStyles>(~static_cast<int>(x)); }
	inline NumberStyles& operator&=(NumberStyles& x, NumberStyles y) { x = x & y; return x; }	
	inline NumberStyles& operator|=(NumberStyles& x, NumberStyles y) { x = x | y; return x; }	
	inline NumberStyles& operator^=(NumberStyles& x, NumberStyles y) { x = x ^ y; return x; }		
	inline bool operator==(const NumberStyles& x, int y) { return static_cast<int>(x) == y; }
	inline bool operator!=(const NumberStyles& x, int y) { return static_cast<int>(x) != y; }	
	**/
	AddFlagSupport(NumberStyles);

	/** Core Parsers **/

	#if (defined(_MSC_VER))
	#pragma warning(push )
	#pragma warning(disable : 4146)		// Disable warning that unary minus operator applied to unsigned type...we know.
	#endif

	template<class T> inline bool Core_Integer_TryParse(const char *pszString, NumberStyles Style, T& Value, int& ParsedCount)
	{
		Value = 0;
		char* pp = (char*)pszString;
		int Base = 10;
		bool Negative = false, SignFound = false;
		ParsedCount = 0;

		if ((Style & NumberStyles::AllowLeadingWhite) != 0)
		{
			while (*pp != 0)
			{
				if (*pp == '\t' || *pp == '\r' || *pp == '\n' || *pp == 0xB || *pp == 0xC || *pp == ' ') pp++;
				else break;
			}
		}

		if ((Style & NumberStyles::AllowLeadingSign) != 0) 
		{
			if (*pp == '+') { pp++; SignFound = true; }
			if (*pp == '-') { pp++; Negative = true; SignFound = true; }
		}

		if ((Style & NumberStyles::AllowHexSpecifier) != 0) Base = 16;
		
		if ((Style & NumberStyles::AllowBasePrefix) != 0)
		{
			if (pp[0] == '0' && tolower(pp[1]) == 'x' ){ Base = 16; pp += 2; }
			else if (tolower(pp[0]) == 'b'){ Base = 2; pp ++; }
		}

		if ((Style & NumberStyles::AllowLeadingSign) != 0)
		{
			if (*pp == '+') { pp++; if (SignFound) return false; }
			if (*pp == '-') { pp++; if (SignFound) return false; Negative = true; }
		}

		if (Base == 16)
		{
			if ((*pp >= '0' && *pp <= '9') || (toupper(*pp) >= 'A' && toupper(*pp) <= 'F'))
			{
				while (*pp)
				{
					if (*pp >= '0' && *pp <= '9')
					{
						Value *= 0x10;
						Value += (*pp - '0');
						pp++;
					}
					else if (*pp >= 'A' && *pp <= 'F')
					{
						Value *= 0x10;
						Value += (*pp - 'A' + 0xA);
						pp++;
					}
					else if (*pp >= 'a' && *pp <= 'f')
					{
						Value *= 0x10;
						Value += (*pp - 'a' + 0xA);
						pp++;
					}
					else if ((Style & NumberStyles::AllowThousands) != 0 && *pp == ',') pp++;
					else break;
				}
				if (Negative) Value = -Value;
				return true;
			}
			return false;
		}
		else if (Base == 10)
		{
			if (*pp >= '0' && *pp <= '9')
			{
				while (*pp)
				{
					if (*pp >= '0' && *pp <= '9')
					{
						Value *= 10;
						Value += (*pp - '0');
						pp++;
					}			
					else if ((Style & NumberStyles::AllowThousands) != 0 && *pp == ',') pp++;
					else break;
				}
				if (Negative) Value = (T)-Value;
				return true;
			}
			return false;
		}
		else if (Base == 2)
		{
			if (*pp == '0' || *pp == '1')
			{
				while (*pp)
				{
					if (*pp == '0') { Value *= 2; pp++; }
					else if (*pp == '1') { Value *= 2; Value ++; pp++; }
					else if ((Style & NumberStyles::AllowThousands) != 0 && *pp == ',') pp++;
					else break;
				}
				if (Negative) Value = (T)-Value;
				return true;
			}
			return false;
		}
		else throw NotSupportedException();
	}

	template<class T> inline bool Core_Float_TryParse(const char *pszString, NumberStyles Style, T& Value, int& ParsedCount)
	{
		Value = 0.0;
		char* pp = (char*)pszString;		
		bool Negative = false;
		ParsedCount = 0;
		T		dFactor = 1.0;

		double  dExpValue = 0;
		bool	bExpNegative = false;
		bool	bExpSignFound = false;		// (True) The +/- in the exponent has been found, or digits have started.
	
		bool	bExponent = false;			// (False) No 'e' char has been detected.  (True) Reading exponent value past the 'e' char.

		if ((Style & NumberStyles::AllowLeadingWhite) != 0)
		{
			while (*pp != 0)
			{
				if (*pp == '\t' || *pp == '\r' || *pp == '\n' || *pp == 0xB || *pp == 0xC || *pp == ' ') pp++;
				else break;
			}
		}

		if ((Style & NumberStyles::AllowLeadingSign) != 0) 
		{
			if (*pp == '+') { pp++; }
			if (*pp == '-') { pp++; Negative = true; }
		}		

		/** Base-10 **/
		
		if ((*pp >= '0' && *pp <= '9') || *pp == '.')
		{
			while (*pp)
			{
				if (!bExponent)
				{
					if (*pp >= '0' && *pp <= '9'){
						if (dFactor < 1.0){ Value += (T)(*pp - '0') * dFactor; dFactor /= 10.0; pp++; }
						else { Value *= 10.0; Value += (T)(*pp - '0'); pp++; }
					}
					else if (*pp == '.') { dFactor = (T)0.1; pp++; }
					else if (*pp == 'e' || *pp == 'E') { bExponent = true; pp++; }
					else if ((Style & NumberStyles::AllowThousands) != 0 && *pp == ',') pp++;
					else break;
				}
				else
				{
					if (*pp >= '0' && *pp <= '9'){
						dExpValue *= 10.0; dExpValue += (double)(*pp - '0');
						bExpSignFound = true;
						pp++;
					}
					else if (!bExpSignFound){
						if (*pp == '+'){ bExpNegative = false; bExpSignFound = true; pp++; }
						else if (*pp == '-'){ bExpNegative = true; bExpSignFound = true; pp++; }
						else break;
					}
					else if ((Style & NumberStyles::AllowThousands) != 0 && *pp == ',') pp++;
					else break;
				}
			}

			if (Negative) Value = -Value;

			if (bExponent)
			{
				if (bExpNegative) dExpValue = -dExpValue;
				Value = (T)(Value * pow((double)10.0, (double)dExpValue));
			}

			return true;
		}
		return false;
	}

	#if (defined(_MSC_VER))
	#pragma warning(pop) 
	#endif

	/** Unsigned integer parsing **/	

	inline bool UInt64_TryParse(const char *pszString, NumberStyles Style, UInt64& Value)
	{
		int ParsedCount;
		return Core_Integer_TryParse<UInt64>(pszString, Style, Value, ParsedCount);
	}

	inline UInt64 UInt64_Parse(const char *pszString, NumberStyles Style)
	{
		UInt64 ret;
		if (!UInt64_TryParse(pszString, Style, ret)) throw FormatException();
		return ret;
	}	

	inline bool UInt32_TryParse(const char* pszString, NumberStyles Style, UInt32& Value)
	{
		UInt64 tmpValue;
		if (UInt64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue > UInt32_MaxValue) return false;
			Value = (UInt32)tmpValue;
			return true;
		}
		return false;
	}

	inline UInt32 UInt32_Parse(const char* pszString, NumberStyles Style)
	{
		UInt64 Value = UInt64_Parse(pszString, Style);
		if (Value > UInt32_MaxValue) throw FormatException(S("Value exceeded 32-bit range."));
		return (UInt32)Value;
	}	

	inline bool UInt16_TryParse(const char* pszString, NumberStyles Style, UInt16& Value)
	{
		UInt64 tmpValue;
		if (UInt64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue > UInt16_MaxValue) return false;
			Value = (UInt16)tmpValue;
			return true;
		}
		return false;
	}

	inline UInt16 UInt16_Parse(const char* pszString, NumberStyles Style)
	{
		UInt64 Value = UInt64_Parse(pszString, Style);
		if (Value > UInt16_MaxValue) throw FormatException(S("Value exceeded 16-bit range."));
		return (UInt16)Value;
	}

	inline bool UInt8_TryParse(const char* pszString, NumberStyles Style, UInt8& Value)
	{
		UInt64 tmpValue;
		if (UInt64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue > UInt8_MaxValue) return false;
			Value = (UInt8)tmpValue;
			return true;
		}
		return false;
	}

	inline UInt8 UInt8_Parse(const char* pszString, NumberStyles Style)
	{
		UInt64 Value = UInt64_Parse(pszString, Style);
		if (Value > UInt8_MaxValue) throw FormatException(S("Value exceeded 8-bit range."));
		return (UInt8)Value;
	}

	inline bool UInt64_TryParse(const string& string, NumberStyles Style, UInt64& Value) { return UInt64_TryParse(string.c_str(), Style, Value); }
	inline UInt64 UInt64_Parse(const string& string, NumberStyles Style) { return UInt64_Parse(string.c_str(), Style); }
	inline bool UInt32_TryParse(const string& string, NumberStyles Style, UInt32& Value) { return UInt32_TryParse(string.c_str(), Style, Value); }
	inline UInt32 UInt32_Parse(const string& string, NumberStyles Style) { return UInt32_Parse(string.c_str(), Style); }
	inline bool UInt16_TryParse(const string& string, NumberStyles Style, UInt16& Value) { return UInt16_TryParse(string.c_str(), Style, Value); }
	inline UInt16 UInt16_Parse(const string& string, NumberStyles Style) { return UInt16_Parse(string.c_str(), Style); }
	inline bool UInt8_TryParse(const string& string, NumberStyles Style, UInt8& Value) { return UInt8_TryParse(string.c_str(), Style, Value); }
	inline UInt8 UInt8_Parse(const string& string, NumberStyles Style) { return UInt8_Parse(string.c_str(), Style); }

	/** Signed integer parsing **/

	inline bool Int64_TryParse(const char *pszString, NumberStyles Style, Int64& Value)
	{
		int ParsedCount;
		return Core_Integer_TryParse<Int64>(pszString, Style, Value, ParsedCount);
	}

	inline Int64 Int64_Parse(const char *pszString, NumberStyles Style)
	{
		Int64 ret;
		if (!Int64_TryParse(pszString, Style, ret)) throw FormatException();
		return ret;
	}	

	inline bool Int32_TryParse(const char* pszString, NumberStyles Style, Int32& Value)
	{
		Int64 tmpValue;
		if (Int64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue < Int32_MinValue || tmpValue > Int32_MaxValue) return false;
			Value = (Int32)tmpValue;
			return true;
		}
		return false;
	}

	inline Int32 Int32_Parse(const char* pszString, NumberStyles Style)
	{
		Int64 Value = Int64_Parse(pszString, Style);
		if (Value < Int32_MinValue || Value > Int32_MaxValue) throw FormatException(S("Value exceeded 32-bit range."));
		return (Int32)Value;
	}

	inline bool Int16_TryParse(const char* pszString, NumberStyles Style, Int16& Value)
	{
		Int64 tmpValue;
		if (Int64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue > Int16_MaxValue) return false;
			Value = (Int16)tmpValue;
			return true;
		}
		return false;
	}

	inline Int16 Int16_Parse(const char* pszString, NumberStyles Style)
	{
		Int64 Value = Int64_Parse(pszString, Style);
		if (Value < Int16_MinValue || Value > Int16_MaxValue) throw FormatException(S("Value exceeded 16-bit range."));
		return (Int16)Value;
	}

	inline bool Int8_TryParse(const char* pszString, NumberStyles Style, Int8& Value)
	{
		Int64 tmpValue;
		if (Int64_TryParse(pszString, Style, tmpValue))
		{ 
			if (tmpValue < Int8_MinValue || tmpValue > Int8_MaxValue) return false;
			Value = (Int8)tmpValue;
			return true;
		}
		return false;
	}

	inline Int8 Int8_Parse(const char* pszString, NumberStyles Style)
	{
		Int64 Value = Int64_Parse(pszString, Style);
		if (Value < Int8_MinValue || Value > Int8_MaxValue) throw FormatException(S("Value exceeded 8-bit range."));
		return (Int8)Value;
	}

	inline bool Int64_TryParse(const string& string, NumberStyles Style, Int64& Value) { return Int64_TryParse(string.c_str(), Style, Value); }
	inline Int64 Int64_Parse(const string& string, NumberStyles Style) { return Int64_Parse(string.c_str(), Style); }
	inline bool Int32_TryParse(const string& string, NumberStyles Style, Int32& Value) { return Int32_TryParse(string.c_str(), Style, Value); }
	inline Int32 Int32_Parse(const string& string, NumberStyles Style) { return Int32_Parse(string.c_str(), Style); }
	inline bool Int16_TryParse(const string& string, NumberStyles Style, Int16& Value) { return Int16_TryParse(string.c_str(), Style, Value); }
	inline Int16 Int16_Parse(const string& string, NumberStyles Style) { return Int16_Parse(string.c_str(), Style); }
	inline bool Int8_TryParse(const string& string, NumberStyles Style, Int8& Value) { return Int8_TryParse(string.c_str(), Style, Value); }
	inline Int8 Int8_Parse(const string& string, NumberStyles Style) { return Int8_Parse(string.c_str(), Style); }

	/** Floating-Point Parsing **/

	inline bool Float_TryParse(const char *pszString, NumberStyles Style, float& Value)
	{
		int ParsedCount;
		return Core_Float_TryParse<float>(pszString, Style, Value, ParsedCount);
	}

	inline bool Double_TryParse(const char *pszString, NumberStyles Style, double& Value)
	{
		int ParsedCount;
		return Core_Float_TryParse<double>(pszString, Style, Value, ParsedCount);
	}

	inline float Float_Parse(const char* pszString, NumberStyles Style)
	{
		float Value;
		if (!Float_TryParse(pszString, Style, Value)) throw FormatException();		
		return Value;
	}

	inline double Double_Parse(const char* pszString, NumberStyles Style)
	{
		double Value;
		if (!Double_TryParse(pszString, Style, Value)) throw FormatException();		
		return Value;
	}

	inline bool Float_TryParse(const string& string, NumberStyles Style, float& Value) { return Float_TryParse(string.c_str(), Style, Value); }
	inline bool Double_TryParse(const string& string, NumberStyles Style, double& Value) { return Double_TryParse(string.c_str(), Style, Value); }
	inline float Float_Parse(const string& string, NumberStyles Style) { return Float_Parse(string.c_str(), Style); }
	inline double Double_Parse(const string& string, NumberStyles Style) { return Double_Parse(string.c_str(), Style); }

	/** Other Parsing **/

	inline bool Bool_TryParse(const char *psz, bool& Value)
	{
		if (compare_no_case(psz, S("true")) == 0 ||
			compare_no_case(psz, S("yes")) == 0 || 
			compare_no_case(psz, S("on")) == 0) { Value = true; return true; }
		if (compare_no_case(psz, S("false")) == 0 ||
			compare_no_case(psz, S("no")) == 0 || 
			compare_no_case(psz, S("off")) == 0) { Value = false; return true; }

		Int32 TmpValue;
		if (Int32_TryParse(psz, NumberStyles::Integer, TmpValue)) 
		{
			if (TmpValue != 0) { Value = true; return true; }
			else { Value = false; return true; }
		}
		else return false;
	}

	inline bool Bool_TryParse(const string& string, bool& Value) { return Bool_TryParse(string.c_str(), Value); }

	inline bool Bool_Parse(const string& string)
	{
		bool Value;
		if (!Bool_TryParse(string, Value)) throw FormatException();
		return Value;
	}
}

#endif	// __BaseTypeParsing_h__

//	End of BaseTypeParsing.h
