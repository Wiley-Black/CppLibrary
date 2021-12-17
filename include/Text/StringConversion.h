/////////
//  StringConversion.h
//  Copyright (C) 2000-2010, 2014 by Wiley Black
//	Author(s):
//		Wiley Black			TheWiley@gmail.com
//
/////////
//	Revision History:
//		August, 2014
//			Merged and organized into new C++ General Library.
//
//		July, 2010
//			Reformed into a conversion-only class to offer similarity to .NET structure.
//
//		November, 2004
//			Renamed to CStringEx.h and .cpp so as to avoid conflicts with the CString.h
//			header included as part of the STL in Borland C++ Builder.
//
//		May 19, 2001
//			Modified CString to support += operations.  This required redefining the
//			whole class, with two different variables (m_nStrAlloc and m_nStrLen) instead
//			of just one, as it had previously been using (m_nStrLen).  Now the class
//			could allocate more memory than needed, keeping a small 'buffer' for
//			adding new data. 
//
//		May 25, 2001
//			Moved many functions to the new CString.cpp file, making them out-of-line.
/////////

#ifndef __WBStringConversion_h__
#define __WBStringConversion_h__

/** Dependencies **/

//#include "../wbFoundation.h"
#include "../Foundation/STL/Text/String.h"
#include "StringHelpers.h"
#include "../Parsing/BaseTypeParsing.h"

/** Content **/

namespace wb
{	
	/** Convert class, offering more detailed conversion capabilities in the .NET styling **/
	// See also to_string() methods offered in String.h.

	class Convert
	{
			// This private version also makes available the # of characters
			// consumed in the reading of the floating-point number.  
		static double ToDouble(const char *psz, int& iIndex, bool& bSuccess);

		static bool IsWhitespace(char ch);
		static bool IsHexDigit(char ch);

		static string Reverse(const string& str);

	public:

			/** Additional Formatting Functions **/

			/** The numeric formatting functions are a little faster than printf() style functions
				since they don't have to parse the format string (%5.6f, for example).  

				nMaxChars will always be the maximum number of characters displayed.  If the
				value to be displayed cannot be shown in 'nMaxChars' characters, the value will 
				be constrained to fit.  For example, if you are trying to display 12345 in 4 
				characters, the number shown will be 9999 instead.  

				Decimal places are the lower priority.  That is, if a number would have to be
				truncated to a smaller number to display it with decimal places, then it will
				not be truncated, fewer or no decimal places will be shown instead.  For example,
				if trying to display 1234.67 in 5 characters, it will be shown as 1234 instead.
				If you are trying to show 1234.67 in 6 characters, it would be shown as 1234.7
				instead.

				'WithCommas' functions operate the same way, adding commas between every 3 digits
				which appear before the decimal place.

				Use 'nMaxChars' of zero to ensure the entire integral value is displayed.  
				The floating-point formatting functions stop when nothing but zeros remain
				to be displayed.
			**/

			/** Double-precision floating-point to string conversions **/
		static string ToString( double dValue, int nMaxChars = 0, int nMaxDecimals = 16, bool bAlwaysShowSign = false );
		static string ToStringWithCommas( double dValue, int nMaxChars = 0, int nMaxDecimals = 16, bool bAlwaysShowSign = false );

			/** Single-precision floating-point to string conversions **/
		static string ToString( float fValue, int nMaxChars = 0, int nMaxDecimals = 8, bool bAlwaysShowSign = false );
		static string ToStringWithCommas( float fValue, int nMaxChars = 0, int nMaxDecimals = 8, bool bAlwaysShowSign = false );

			/** Integer to string conversions **/
		static string ToString( int nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );
		static string ToStringWithCommas( int nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );
		static string ToString( Int64 nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );

			/** Unsigned Integer to string conversions **/
		static string ToString( unsigned int nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );
		static string ToStringWithCommas( unsigned int nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );	
		static string ToString( UInt64 nValue, int nMaxChars = 0, bool bAlwaysShowSign = false );
	
			/** Boolean to string conversion **/
		static const char *ToString(bool Value);

			/** Computer Data Size to string conversion (i.e. 2097152 -> "2 MiB") **/
		static string ToDataSizeString(UInt64 DataSize, int Digits = 1);
	};

	/** Implementation **/

	inline string Convert::Reverse(const string& str)
	{
		string ret;
		for (size_t ii = str.length() - 1; ii > 0; ii--) ret += str[ii];
		if (str.length() > 0) ret += str[0];
		return ret;
	}

	inline string Convert::ToString(float fValue, int nMaxChars, int nMaxDecimals, bool bAlwaysShowSign){
		return ToString(double(fValue), nMaxChars, nMaxDecimals, bAlwaysShowSign);
	}

	inline string Convert::ToStringWithCommas( float fValue, int nMaxChars, int nMaxDecimals, bool bAlwaysShowSign ){
		return ToStringWithCommas(double(fValue), nMaxChars, nMaxDecimals, bAlwaysShowSign);
	}

	inline const char *Convert::ToString(bool Value) { if (Value) return "true"; else return "false"; }	

	inline bool Convert::IsWhitespace(char ch){ return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r'; }
	inline bool Convert::IsHexDigit(char ch){ return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'); }	

	inline string Convert::ToStringWithCommas(double dValue, int nChars /*= 0*/, int nMaxDecimals /*= 20*/, bool bAlwaysShowSign /*= false*/)
	{
		string ret;
		int ii;

			// This algorithm generates the string in reverse, and the last step is to reverse the string.

		if( nChars > 60 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);
		bool bNegative = (dValue < 0.0);

		if( bNegative ){
			if( nChars ) nChars --;
			dValue = -dValue;
		}
		else if( bAlwaysShowSign && nChars ) nChars --;

		int nCommas = 0;
		{
			double dValueIter = dValue;
			for( ii = 0; dValueIter >= 1.0; ii++, dValueIter /= 10.0 ) 
				if( ii && !(ii % 3) ) nCommas ++;
		}

		if( !bUnlimitedChars )
		{
			double dMaxValue = 0.99999999999999999999999999999999;
			for( ii = 0; ii < nChars - nCommas; ii++ ) dMaxValue *= 10.0;
			double dMaxValueWith1Decimal = dMaxValue / 100.0;
			if( dValue >= dMaxValue ) dValue = dMaxValue;
			if( dValue >= dMaxValueWith1Decimal 
			 || dMaxValueWith1Decimal < 1.0 ) nMaxDecimals = 0;
	
			if( nMaxDecimals ){
					// Loop, reducing nMaxDecimals until we have few enough decimal places for the non-decimal
					// part to fit alright.  Keeping in mind that nMaxValueWithNDecimal is the maximum value with
					// N decimal places *based on the maximum number of characters allowed, nChars*.  We already
					// know that at least 1 decimal will fit, so we begin there.  That is also why we reduce
					// 'dMaxValueWith1Decimal' before the if() check, and why we pre-decrement nMaxDecimals.
				int nUseDecimals = 1;
				while( -- nMaxDecimals ){
					dMaxValueWith1Decimal /= 10.0;		// Now 'MaxValueWith N Decimals'.  e.g. 9.99
					if( dValue >= dMaxValueWith1Decimal				// e.g. if( 10000.0 >= 9.99 ) break
					 || dMaxValueWith1Decimal < 1.0 ) break; else nUseDecimals ++; 
				}
				nMaxDecimals = nUseDecimals;
			}
		}

		dValue *= pow( 10.0, double(nMaxDecimals) );

		if( fmod( dValue, 1.0 ) >= 0.5 ) dValue += 1.0;			// Round upward	

		for( ii = 1; nChars || bUnlimitedChars; )
		{
			uint nDigit = (uint)fmod( dValue, 10.0 );			// Note: Integer conversion always rounds down.
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			dValue /= 10.0;
			if( nMaxDecimals ){
				nMaxDecimals --;
				if( !nMaxDecimals ){				
					ret += S(".");
					if( nChars ) nChars --;						// Count off for the decimal place
					if( dValue < 1.0 ){							// If nothing above the decimal, make sure					
						ret += S("0");							// that a zero gets shown before the decimal.
						if( nChars ) nChars --;
						break;
					}
				}
			}
			else {
				if( dValue < 1.0 ) break;						// No more characters to show (except zeros)
				if( !(ii % 3) ){
					ret += S(",");
					if( nChars ) nChars --;
				}
				ii++;
			}
		}

		if( bNegative ) ret += S("-");
		else if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

		/**
			ToString() converts a double-precision floating-point number to a string.
		**/
	inline string Convert::ToString(double dValue, int nChars /*= 0*/, int nMaxDecimals /*= 20*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

			// This algorithm generates the string in reverse, and the last step is to reverse the string.

		if( nChars > 50 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);
		bool bNegative = (dValue < 0.0);

		string ret;

		if( bNegative ){
			if( nChars ) nChars --;
			dValue = -dValue;
		}
		else if( bAlwaysShowSign && nChars ) nChars --;

		if( !bUnlimitedChars )
		{
			double dMaxValue = 0.99999999999999999999999999999999;
			for( ii = 0; ii < nChars; ii++ ) dMaxValue *= 10.0;
			double dMaxValueWith1Decimal = dMaxValue / 100.0;
			if( dValue >= dMaxValue ) dValue = dMaxValue;				// e.g. if( 10000.0 >= 9999.99 ) use 9999.99.
			if( dValue >= dMaxValueWith1Decimal							// e.g. if( 10000.0 >= 99.99 ) no decimals.
			 || dMaxValueWith1Decimal < 1.0 ) nMaxDecimals = 0;	

			if( nMaxDecimals ){
					// Loop, reducing nMaxDecimals until we have few enough decimal places for the non-decimal
					// part to fit alright.  Keeping in mind that nMaxValueWithNDecimal is the maximum value with
					// N decimal places *based on the maximum number of characters allowed, nChars*.  We already
					// know that at least 1 decimal will fit, so we begin there.  That is also why we reduce
					// 'dMaxValueWith1Decimal' before the if() check, and why we pre-decrement nMaxDecimals.
				int nUseDecimals = 1;
				while( -- nMaxDecimals ){
					dMaxValueWith1Decimal /= 10.0;		// Now 'MaxValueWith N Decimals'.  e.g. 9.99
					if( dValue >= dMaxValueWith1Decimal				// e.g. if( 10000.0 >= 9.99 ) break
					 || dMaxValueWith1Decimal < 1.0 ) break; else nUseDecimals ++; 
				}
				nMaxDecimals = nUseDecimals;
			}
		}

		dValue *= pow( 10.0, double(nMaxDecimals) );

		if( fmod( dValue, 1.0 ) >= 0.5 ) dValue += 1.0;			// Round upward	

		while( nChars || bUnlimitedChars )
		{
			uint nDigit = (uint)fmod( dValue, 10.0 );			// Note: Integer conversion always rounds down.
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			dValue /= 10.0;
			if( nMaxDecimals ){
				nMaxDecimals --;
				if( !nMaxDecimals ){
					ret += S(".");
					if( nChars ) nChars --;						// Count off for the decimal place
					if( dValue < 1.0 ){							// If nothing above the decimal, make sure
						ret += S("0");							// that a zero gets shown before the decimal.
						if( nChars ) nChars --;
						break;
					}
				}
			}
			else if( dValue < 1.0 ) break;						// No more characters to show (except zeros)
		}

		if( bNegative ) ret += S("-");
		else if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

	inline string Convert::ToStringWithCommas(int nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 60 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);
		bool bNegative = (nValue < 0);

		string ret;	

		if( bNegative ){
			if( nChars ) nChars --;
			nValue = -nValue;
		}
		else if( bAlwaysShowSign && nChars ) nChars --;

		int nCommas = 0;
		{
			int nValueIter = nValue;
			for( ii = 0; nValueIter >= 1; ii++, nValueIter /= 10 )
				if( ii && !(ii % 3) ) nCommas ++;
		}

		if( !bUnlimitedChars )
		{
			int nMaxValue = 0;
			for( ii = 0; ii < nChars - nCommas; ii++ ) nMaxValue = (nMaxValue * 10) + 9;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		for( ii = 1; nChars || bUnlimitedChars; ii ++ )
		{
			int nDigit = nValue % 10;
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10;
			if( !nValue ) break;						// No more characters to show (except zeros)
			if( !(ii % 3) ){
				ret += S(",");
				if( nChars ) nChars --;
			}
		}

		if( bNegative ) ret += S("-");
		else if( bAlwaysShowSign ) ret += S("+");

		while (nChars --) ret += S(" ");

		return Reverse(ret);
	}

		/**
			ToString() converts a integer number to a string.
		**/
	inline string Convert::ToString(int nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 50 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);
		bool bNegative = (nValue < 0);

		string ret;	

		if( bNegative ){
			if( nChars ) nChars --;
			nValue = -nValue;
		}
		else if( bAlwaysShowSign && nChars ) nChars --;

		if( !bUnlimitedChars )
		{
			int nMaxValue = 0;
			for( ii = 0; ii < nChars; ii++ ) nMaxValue = (nMaxValue * 10) + 9;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		while( nChars || bUnlimitedChars )
		{
			int nDigit = nValue % 10;
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10;
			if( !nValue ) break;						// No more characters to show (except zeros)
		}

		if( bNegative ) ret += S("-");
		else if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

	inline string Convert::ToStringWithCommas(unsigned int nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 60 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);	

		string ret;

		if (bAlwaysShowSign && nChars) nChars --;

		int nCommas = 0;
		{
			unsigned int nValueIter = nValue;
			for( ii = 0; nValueIter >= 1; ii++, nValueIter /= 10U )
				if( ii && !(ii % 3U) ) nCommas ++;
		}

		if( !bUnlimitedChars )
		{
			unsigned int nMaxValue = 0;
			for( ii = 0; ii < nChars - nCommas; ii++ ) nMaxValue = (nMaxValue * 10U) + 9U;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		for( ii = 1; nChars || bUnlimitedChars; ii ++ )
		{
			unsigned int nDigit = nValue % 10U;
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10U;
			if( !nValue ) break;						// No more characters to show (except zeros)
			if( !(ii % 3) ){
				ret += S(",");
				if( nChars ) nChars --;
			}
		}

		if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

		/**
			ToString() converts a integer number to a string.
		**/
	inline string Convert::ToString(unsigned int nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 50 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);

		string ret;	

		if( bAlwaysShowSign && nChars ) nChars --;

		if( !bUnlimitedChars )
		{
			unsigned int nMaxValue = 0;
			for( ii = 0; ii < nChars; ii++ ) nMaxValue = (nMaxValue * 10U) + 9U;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		while( nChars || bUnlimitedChars )
		{
			unsigned int nDigit = nValue % 10U;
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10U;
			if( !nValue ) break;						// No more characters to show (except zeros)
		}

		if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

		/**
			ToString() converts a integer number to a string.
		**/
	inline string Convert::ToString(Int64 nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 50 ) nChars = 0;		// Signed 64-bit integers can occupy up to 20 decimal digits.

		bool bUnlimitedChars = (nChars == 0);
		bool bNegative = (nValue < 0);

		string ret;	

		if( bNegative ){
			if( nChars ) nChars --;
			nValue = -nValue;
		}
		else if( bAlwaysShowSign && nChars ) nChars --;

		if( !bUnlimitedChars )
		{
			Int64 nMaxValue = 0;
			for( ii = 0; ii < nChars; ii++ ) nMaxValue = (nMaxValue * 10) + 9;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		while( nChars || bUnlimitedChars )
		{
			int nDigit = (int)(nValue % 10ll);
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10ll;
			if( !nValue ) break;						// No more characters to show (except zeros)
		}

		if( bNegative ) ret += S("-");
		else if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}

		/**
			ToString() converts a integer number to a string.
		**/
	inline string Convert::ToString(UInt64 nValue, int nChars /*= 0*/, bool bAlwaysShowSign /*= false*/)
	{
		int ii;

		if( nChars > 50 ) nChars = 0;

		bool bUnlimitedChars = (nChars == 0);

		string ret;	

		if( bAlwaysShowSign && nChars ) nChars --;

		if( !bUnlimitedChars )
		{
			UInt64 nMaxValue = 0ull;
			for( ii = 0; ii < nChars; ii++ ) nMaxValue = (nMaxValue * 10ull) + 9ull;
			if( nValue >= nMaxValue ) nValue = nMaxValue;
		}

		while( nChars || bUnlimitedChars )
		{
			unsigned int nDigit = (unsigned int)(nValue % 10ull);
			ret += (char)(nDigit + '0');
			if( nChars ) nChars --;
			nValue /= 10ull;
			if( !nValue ) break;						// No more characters to show (except zeros)
		}

		if( bAlwaysShowSign ) ret += S("+");

		while( nChars -- ) ret += S(" ");

		return Reverse(ret);
	}	

	inline string Convert::ToDataSizeString(UInt64 DataSize, int Digits)
	{
		constexpr UInt64 TiB = 1099511627776ull;
		constexpr UInt64 GiB = 1073741824ull;
		constexpr UInt64 MiB = 1048576ull;
		constexpr UInt64 KiB = 1024ull;

		if (Digits > 1)
		{
			int MaxDec = Digits - 1;
			if (DataSize > (4 * TiB))
				return ToStringWithCommas((float)DataSize / TiB, 0, MaxDec) + " TiB";
			if (DataSize > (4 * GiB))
				return ToStringWithCommas((float)DataSize / GiB, 0, MaxDec) + " GiB";
			if (DataSize > (4 * MiB))
				return ToStringWithCommas((float)DataSize / MiB, 0, MaxDec) + " MiB";
			if (DataSize > (4 * KiB))
				return ToStringWithCommas((float)DataSize / KiB, 0, MaxDec) + " KiB";
			return ToStringWithCommas((UInt32)DataSize) + " bytes";
		}

		if (DataSize > (4 * TiB))
			return ToString((int)Round((float)DataSize / TiB)) + " TiB";
		if (DataSize > (4 * GiB))
			return ToString((int)Round((float)DataSize / GiB)) + " GiB";
		if (DataSize > (4 * MiB))
			return ToString((int)Round((float)DataSize / MiB)) + " MiB";
		if (DataSize > (4 * KiB))
			return ToString((int)Round((float)DataSize / KiB)) + " KiB";
		return ToString(DataSize) + " bytes";
	}

	#if 0
	GUID CStringEx::asGUID() const
	{
		CGUID cret;
		if( !cret.fromString(*this) ) cret.Zero();
		return cret;
	}

	complex<double>	CStringEx::asComplex() const
	{
		int iDivider;
		double dFirst = asDouble(iDivider);

		bool bFirstReal = true;
		for( ; iDivider < GetLength(); iDivider ++ )
		{
			if( toupper(GetAt(iDivider)) == 'I' || toupper(GetAt(iDivider)) == 'J' ) bFirstReal = false;
			else if( IsWhitespace(GetAt(iDivider)) ) continue;
			else break;
		}

		CStringEx	strSecond = Mid(iDivider);

		int iSecondDivider;
		double dSecond = strSecond.asDouble(iSecondDivider);
		bool bSecondReal = true;
		iDivider += iSecondDivider;
		for( ; iDivider < GetLength(); iDivider ++ )
		{
			if( toupper(GetAt(iDivider)) == 'I' || toupper(GetAt(iDivider)) == 'J' ) bSecondReal = false;
			else if( IsWhitespace(GetAt(iDivider)) ) continue;
			else break;
		}

		if( bFirstReal && !bSecondReal ){
			complex<double> cRet( dFirst, dSecond );
			return cRet;
		}
		else if( bSecondReal && !bFirstReal ) return complex<double>( dSecond, dFirst );
		else if( fabs(dSecond) < 1.0e-100 ) return bFirstReal ? complex<double>( dFirst, 0.0 ) : complex<double>( 0.0, dFirst );
		else return complex<double>( dFirst, dSecond );
	}
	#endif

}

#endif  // __StringConversion_h__

//  End of StringConversion.h

