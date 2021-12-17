/**	TimeSpan.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
**/

/*	A class for representing elapsed times.

	The time is stored in the seconds.  It represents the *actual*
	time elapsed.  Effects such as leap years are not represented in
	the time span, however, this class interacts with the DateTime
	class to incorporate such effects.
*/
/////////

#ifndef __TimeSpan_h__
#define __TimeSpan_h__

#include <math.h>
#include <stdlib.h>
#include <assert.h>
#include "../Foundation/STL/Text/String.h"
#include "../Foundation/Exceptions.h"
#include "../Foundation/STL/Text/String.h"
#include "../Parsing/BaseTypeParsing.h"
#include "TimeConstants.h"

namespace wb
{
	class TimeSpan
	{
	protected:

		/** If m_nElapsedSeconds is negative, then m_nElapsedNanoseconds will be stored negative as well.  The two would be "added" to
			accomplish the total time.  Use IsNegative() to check for negative spans, since m_nElapsedSeconds could be zero when
			m_nElapsedNanoseconds is non-zero negative. **/

		Int64	m_nElapsedSeconds;
		Int32	m_nElapsedNanoseconds;		

		friend class DateTime;

	public:

		/** TimeSpan() constructors
			Parameters do not have constraints.  I.e., minutes do not need to be 1-60.  This allows
			the creation of an object such as:
				TimeSpan( 0, 48, 0, 0 ) for 48 hours.
		**/

		TimeSpan();
		TimeSpan( const TimeSpan& );		
		TimeSpan( Int64 nDays, int nHours, int nMinutes, int nSeconds, int nNanoseconds = 0 );
	#	ifdef _MFC
		TimeSpan( const CTimeSpan& );
	#	endif

		static TimeSpan FromSeconds(Int64 nElapsedSeconds, Int32 nElapsedNanoseconds = 0);
		static TimeSpan FromSeconds(double ElapsedSeconds);
		static TimeSpan FromNanoseconds(Int64 nElapsedNanoseconds);

		static TimeSpan GetInvalid();
		static TimeSpan GetZero();
		/*
		const TimeSpan Invalid;
		const TimeSpan Zero;
		*/

		// All Get...() calls (except GetTotal...() and GetApproxTotal...() calls) will return the absolute time.  For example, if 
		// the time span represents -2 hours, then GetDays() = 0, GetHours() = 2, so forth, and IsNegative() is true.

			// Get...() calls return the number of units, less any larger units, rounded downward.
			// For example:  119 seconds -> GetMinutes() = 1, GetSeconds() = 59.			
		bool	IsNegative() const;
		Int64	GetDays() const;
		Int32	GetHours() const;
		Int32	GetMinutes() const;
		Int32	GetSeconds() const;
		Int32	GetNanoseconds() const;

			// This call is slightly faster than the individual Get...() calls.
		void	Get( Int64& nDays, Int32& nHours, Int32& nMinutes, Int32& nSeconds ) const;
		void	Get( Int64& nDays, Int32& nHours, Int32& nMinutes, Int32& nSeconds, Int32& nNanoseconds ) const;

			// Note: The GetTotal...() functions round to the nearest whole number.  For example, 119 seconds -> 2 minutes.
		Int64	GetTotalDays() const;
		Int64	GetTotalHours() const;
		Int64	GetTotalMinutes() const;
		Int64	GetTotalSeconds() const;

			// These GetTotal...() functions will throw an exception if the returned value cannot fit in an Int64 value.  For
			// nanoseconds this occurs with intervals longer than about 290 years, and is longer for the remaining calls.
		Int64	GetTotalMilliseconds() const;
		Int64	GetTotalMicroseconds() const;

			// These GetTotal...() functions use no rounding.  GetTotalNanoseconds will throw an exception on intervals longer
			// than about 290 years.
		Int64	GetTotalSecondsNoRounding() const;
		Int64	GetTotalNanoseconds() const;

			// Since CDateTimeSpan objects are not affixed to a calender date, these functions provide approximations only.	
			// Get..Total..() calls are rounded.  Other Get..() calls always round down.
		Int64	GetApproxYears() const;
		Int32	GetApproxMonths() const;
		Int32	GetApproxDays() const;
		Int64	GetApproxTotalYears() const;
		Int64	GetApproxTotalMonths() const;	

			/** fromString()
				Attempts to parse a string into an elapsed time.  Returns Invalid if a valid format was not
				recognized and parsed successfully.  The following are examples of supported formats:

					HH:MM:SS
					MM:SS
					SS
					HH:MM:SS text
					MM:SS text
					SS text

				where:
					SS represents seconds, and may contain any number of decimal digits.
			**/
		static bool TryParse(const char*, TimeSpan&);
		static TimeSpan Parse(const char*);

			/** asExactString()
				Returns string as "XX days H:MM:SS.sss hours", always including seconds.
				Returns string as "H:MM:SS.sss hours", when less than a day elapsed.

				If Precision is zero then the ".sss" component is omitted.  Otherwise Precision controls the number
				of digits after the decimal place, up to a maximum of 9.
			**/

		string ToString(int Precision = 9) const;
		string ToShortString() const;		// Returns string as "YY months", showing only highest-level unit.

			/** Operations **/

		bool operator==( const TimeSpan& ) const;
		bool operator!=( const TimeSpan& ) const;
		bool operator<=( const TimeSpan& ) const;
		bool operator<( const TimeSpan& ) const;
		bool operator>=( const TimeSpan& ) const;
		bool operator>( const TimeSpan& ) const;

		TimeSpan operator+( const TimeSpan& ) const;
		TimeSpan operator-( const TimeSpan& ) const;

		const TimeSpan& operator=( const TimeSpan& );
	};

	/** Non-class operations **/

	inline string to_string(const TimeSpan& Value) { return Value.ToString(); }

	/////////
	//	TimeSpan Constants
	//

	inline /*static*/ TimeSpan TimeSpan::GetInvalid() {
		TimeSpan ret;
		ret.m_nElapsedSeconds = Int64_MaxValue;
		ret.m_nElapsedNanoseconds = Int32_MaxValue;
		return ret;
	}
	inline /*static*/ TimeSpan TimeSpan::GetZero() {
		TimeSpan ret;
		ret.m_nElapsedSeconds = 0;
		ret.m_nElapsedNanoseconds = 0;
		return ret;
	}

	#if 0	// Syntactically nice, but harder in header-only format.
	/*static*/ const TimeSpan TimeSpan::Invalid = TimeSpan::GetInvalid();
	/*static*/ const TimeSpan TimeSpan::Zero = TimeSpan::GetZero();
	#endif

	/////////
	//  Inline functions
	//

	inline TimeSpan::TimeSpan() { m_nElapsedSeconds = 0; m_nElapsedNanoseconds = 0; }
	inline TimeSpan::TimeSpan( const TimeSpan& cp ) : m_nElapsedSeconds(cp.m_nElapsedSeconds), m_nElapsedNanoseconds(cp.m_nElapsedNanoseconds) { }	
	inline TimeSpan::TimeSpan(Int64 nDays, int nHours, int nMinutes, int nSeconds, int nNanoseconds /*= 0*/) {
		m_nElapsedSeconds =  (nDays * time_constants::g_nSecondsPerDay) + ((Int64)nHours * time_constants::g_nSecondsPerHour) 
						  + ((Int64)nMinutes * time_constants::g_nSecondsPerMinute) + (Int64)nSeconds;
		m_nElapsedNanoseconds = abs(nNanoseconds);
		if (m_nElapsedSeconds < 0) m_nElapsedNanoseconds = -m_nElapsedNanoseconds;
		assert (m_nElapsedNanoseconds > -1000000000 && m_nElapsedNanoseconds < 1000000000);
	}
	inline /*static*/ TimeSpan TimeSpan::FromSeconds(Int64 nElapsedSeconds, Int32 nElapsedNanoseconds /*= 0*/) {
		TimeSpan ret;
		ret.m_nElapsedSeconds = nElapsedSeconds;
		ret.m_nElapsedNanoseconds = nElapsedNanoseconds;
		assert((ret.m_nElapsedNanoseconds > -1000000000 && ret.m_nElapsedNanoseconds < 1000000000)
			|| (ret.m_nElapsedSeconds == Int64_MaxValue && ret.m_nElapsedNanoseconds == Int32_MaxValue));
		return ret;
	}
	inline /*static*/ TimeSpan TimeSpan::FromSeconds(double ElapsedSeconds) {
		Int64 Whole = (Int64)floor(ElapsedSeconds);
		double Rem = fmod(ElapsedSeconds, 1.0);
		Int32 Nanoseconds = Round32(1000000000.0 * Rem);
		return FromSeconds(Whole, Nanoseconds);
	}
	inline /*static*/ TimeSpan TimeSpan::FromNanoseconds(Int64 nElapsedNanoseconds) {
		TimeSpan ret;
		ret.m_nElapsedSeconds = nElapsedNanoseconds / 1000000000ll;
		ret.m_nElapsedNanoseconds = (Int32)(nElapsedNanoseconds % 1000000000ll);
		return ret;
	}
	#ifdef _MFC
	inline TimeSpan::TimeSpan( const CTimeSpan& tmSpan ){ m_nElapsedSeconds = tmSpan.GetTotalSeconds(); m_nElapsedNanoseconds = 0; }
	#endif	

	/** If less than one second is stored, then m_nElapsedSeconds will be zero but m_nElapsedNanoseconds can still be negative.  Therefore,
		we must check both. **/
	inline bool		TimeSpan::IsNegative() const { return (m_nElapsedSeconds < 0 || m_nElapsedNanoseconds < 0); }

	inline Int64	TimeSpan::GetDays() const {
			// GetDays() is the same as GetTotalDays() except that this version rounds downward.
		return abs(m_nElapsedSeconds) /*seconds*/ / time_constants::g_nSecondsPerDay /*seconds/day*/;
	}

	inline Int32	TimeSpan::GetHours() const { return (Int32)(abs(m_nElapsedSeconds) % time_constants::g_nSecondsPerDay) / time_constants::g_n32SecondsPerHour; }
	inline Int32	TimeSpan::GetMinutes() const { return (Int32)(abs(m_nElapsedSeconds) % time_constants::g_nSecondsPerHour) / time_constants::g_n32SecondsPerMinute; }
	inline Int32	TimeSpan::GetSeconds() const { return (Int32)(abs(m_nElapsedSeconds) % time_constants::g_nSecondsPerMinute); }	
	inline Int32	TimeSpan::GetNanoseconds() const { return abs(m_nElapsedNanoseconds); }	
	
	// Note: We disregard the nanoseconds in the following calculations.  The only effect is to nudge the rounding by 0.5 seconds, which only matters if
	// the total days, hours, or minutes value was directly on a half-unit.
	inline Int64	TimeSpan::GetTotalDays() const { return Round64(m_nElapsedSeconds /*seconds*/ / time_constants::g_dSecondsPerDay /*seconds/day*/ ); }
	inline Int64	TimeSpan::GetTotalHours() const { return Round64(m_nElapsedSeconds /*seconds*/ / time_constants::g_dSecondsPerHour /*seconds/hour*/ ); }
	inline Int64	TimeSpan::GetTotalMinutes() const { return Round64(m_nElapsedSeconds /*seconds*/ / time_constants::g_dSecondsPerMinute /*seconds/minute*/ ); }

	// Now we consider nanoseconds...
	inline Int64	TimeSpan::GetTotalSeconds() const { 
		if (m_nElapsedSeconds >= 0)
			return (m_nElapsedNanoseconds > (time_constants::g_n32NanosecondsPerSecond/2)) ? (m_nElapsedSeconds + 1) : (m_nElapsedSeconds);
		else
			return (m_nElapsedNanoseconds > -(time_constants::g_n32NanosecondsPerSecond/2)) ? (m_nElapsedSeconds) : (m_nElapsedSeconds - 1);
	}

	inline Int64	TimeSpan::GetTotalSecondsNoRounding() const { return m_nElapsedSeconds; }

	inline Int64	TimeSpan::GetTotalMilliseconds() const { 
		static const Int64 MSPerS = 1000ll;
		static const Int64 MaxS = (Int64_MaxValue / MSPerS);
		if (abs(m_nElapsedSeconds) + 1 > MaxS)
			throw ArgumentOutOfRangeException("Cannot retrieve total milliseconds on time spans longer than " + std::to_string(MaxS) + " seconds.");
		if (m_nElapsedSeconds >= 0)
			return (m_nElapsedSeconds * MSPerS) + Round64(m_nElapsedNanoseconds / 1000000.0);
		else 
			return (m_nElapsedSeconds * MSPerS) - Round64(m_nElapsedNanoseconds / 1000000.0);
	}

	inline Int64	TimeSpan::GetTotalMicroseconds() const { 
		static const Int64 USPerS = 1000000ll;
		static const Int64 MaxS = (Int64_MaxValue / USPerS);
		if (abs(m_nElapsedSeconds) + 1 > MaxS)
			throw ArgumentOutOfRangeException("Cannot retrieve total microseconds on time spans longer than " + std::to_string(MaxS) + " seconds.");
		if (m_nElapsedSeconds >= 0)
			return (m_nElapsedSeconds * USPerS) + Round64(m_nElapsedNanoseconds / 1000.0);
		else 
			return (m_nElapsedSeconds * USPerS) - Round64(m_nElapsedNanoseconds / 1000.0);
	}

	inline Int64	TimeSpan::GetTotalNanoseconds() const { 
		static const Int64 NSPerS = 1000000000ll;
		static const Int64 MaxS = (Int64_MaxValue / NSPerS);
		if (abs(m_nElapsedSeconds) + 1 > MaxS)
			throw ArgumentOutOfRangeException("Cannot retrieve total nanoseconds on time spans longer than " + to_string(MaxS) + " seconds.");
		if (m_nElapsedSeconds >= 0)
			return (m_nElapsedSeconds * NSPerS) + (Int64)m_nElapsedNanoseconds;
		else 
			return (m_nElapsedSeconds * NSPerS) - (Int64)m_nElapsedNanoseconds;
	}

			// Since TimeSpan objects are not affixed to a calender date, these functions provide approximations only.

	inline Int64	TimeSpan::GetApproxTotalYears() const { return Round64(m_nElapsedSeconds /*seconds*/ / time_constants::g_dApproxSecondsPerYear /*seconds/year*/ ); }
	inline Int64	TimeSpan::GetApproxTotalMonths() const { return Round64(m_nElapsedSeconds /*seconds*/ / time_constants::g_dApproxSecondsPerMonth /*seconds/month*/ ); }

	inline Int64	TimeSpan::GetApproxYears() const { return abs(m_nElapsedSeconds) /*seconds*/ / time_constants::g_nApproxSecondsPerYear /*seconds/year*/; }
	inline Int32	TimeSpan::GetApproxMonths() const { return (Int32)(abs(m_nElapsedSeconds) % time_constants::g_nApproxSecondsPerYear) / time_constants::g_n32ApproxSecondsPerMonth; }
	inline Int32	TimeSpan::GetApproxDays() const { return (Int32)(abs(m_nElapsedSeconds) % time_constants::g_nApproxSecondsPerMonth) / time_constants::g_n32SecondsPerDay; }

	inline void		TimeSpan::Get(Int64& nDays, Int32& nHours, Int32& nMinutes, Int32& nSeconds) const 
	{
			// This function uses the fast that multiplication is slightly faster than division as an optimization technique.
			// It also is slightly faster because less 64-bit arithmetic is required having retained the remainders.
		nDays = abs(m_nElapsedSeconds) /*seconds*/ / time_constants::g_nSecondsPerDay /*seconds/day*/;
		// assert(abs( (Int64)(m_nElapsedSeconds - (nDays * time_constants::g_nSecondsPerDay)) ) < Int32_MaxValue);
		Int32 nRemainder = (Int32)(abs(m_nElapsedSeconds) - (nDays * time_constants::g_nSecondsPerDay));

		assert( nRemainder < time_constants::g_nSecondsPerDay );
		nHours = nRemainder / time_constants::g_n32SecondsPerHour;
		nRemainder -= (nHours * time_constants::g_n32SecondsPerHour);

		assert( nRemainder < time_constants::g_n32SecondsPerHour );
		nMinutes = nRemainder / time_constants::g_n32SecondsPerMinute;
		nRemainder -= (nMinutes * time_constants::g_n32SecondsPerMinute);

		assert( nRemainder < time_constants::g_n32SecondsPerMinute );
		nSeconds = nRemainder;

	}// End of Get()

	inline void		TimeSpan::Get(Int64& nDays, Int32& nHours, Int32& nMinutes, Int32& nSeconds, Int32& nNanoseconds) const 
	{
		Get(nDays, nHours, nMinutes, nSeconds);
		nNanoseconds = abs(m_nElapsedNanoseconds);
	}			

	inline bool TimeSpan::operator==( const TimeSpan& span ) const { return m_nElapsedSeconds == span.m_nElapsedSeconds && m_nElapsedNanoseconds == span.m_nElapsedNanoseconds; }
	inline bool TimeSpan::operator!=( const TimeSpan& span ) const { return m_nElapsedSeconds != span.m_nElapsedSeconds || m_nElapsedNanoseconds != span.m_nElapsedNanoseconds; }
	inline bool TimeSpan::operator<=( const TimeSpan& span ) const { return m_nElapsedSeconds < span.m_nElapsedSeconds || (m_nElapsedSeconds == span.m_nElapsedSeconds && m_nElapsedNanoseconds <= span.m_nElapsedNanoseconds); }
	inline bool TimeSpan::operator<( const TimeSpan& span ) const { return m_nElapsedSeconds < span.m_nElapsedSeconds || (m_nElapsedSeconds == span.m_nElapsedSeconds && m_nElapsedNanoseconds < span.m_nElapsedNanoseconds); }
	inline bool TimeSpan::operator>=( const TimeSpan& span ) const { return m_nElapsedSeconds > span.m_nElapsedSeconds || (m_nElapsedSeconds == span.m_nElapsedSeconds && m_nElapsedNanoseconds >= span.m_nElapsedNanoseconds); }
	inline bool TimeSpan::operator>( const TimeSpan& span ) const { return m_nElapsedSeconds > span.m_nElapsedSeconds || (m_nElapsedSeconds == span.m_nElapsedSeconds && m_nElapsedNanoseconds > span.m_nElapsedNanoseconds); }	

	inline TimeSpan TimeSpan::operator+( const TimeSpan& span ) const 
	{ 
		Int32 NewNanoseconds = m_nElapsedNanoseconds + span.m_nElapsedNanoseconds;
		if (NewNanoseconds >= time_constants::g_n32NanosecondsPerSecond) return TimeSpan::FromSeconds(m_nElapsedSeconds + span.m_nElapsedSeconds + 1, NewNanoseconds - time_constants::g_n32NanosecondsPerSecond);
		else if (NewNanoseconds <= -time_constants::g_n32NanosecondsPerSecond) return TimeSpan::FromSeconds(m_nElapsedSeconds + span.m_nElapsedSeconds - 1, NewNanoseconds + time_constants::g_n32NanosecondsPerSecond);
		else return TimeSpan::FromSeconds(m_nElapsedSeconds + span.m_nElapsedSeconds, NewNanoseconds); 
	}
	inline TimeSpan TimeSpan::operator-( const TimeSpan& span ) const 
	{ 
		Int32 NewNanoseconds = m_nElapsedNanoseconds - span.m_nElapsedNanoseconds;
		if (NewNanoseconds >= time_constants::g_n32NanosecondsPerSecond) return TimeSpan::FromSeconds(m_nElapsedSeconds - span.m_nElapsedSeconds + 1, NewNanoseconds - time_constants::g_n32NanosecondsPerSecond);
		else if (NewNanoseconds <= -time_constants::g_n32NanosecondsPerSecond) return TimeSpan::FromSeconds(m_nElapsedSeconds - span.m_nElapsedSeconds - 1, NewNanoseconds + time_constants::g_n32NanosecondsPerSecond);
		else return TimeSpan::FromSeconds(m_nElapsedSeconds - span.m_nElapsedSeconds, NewNanoseconds); 
	}

	inline const TimeSpan& TimeSpan::operator=( const TimeSpan& cp ){ m_nElapsedSeconds = cp.m_nElapsedSeconds; m_nElapsedNanoseconds = cp.m_nElapsedNanoseconds; return *this; }

		/** Parsing Routines **/

	inline bool TimeSpan::TryParse(const char* lpsz, TimeSpan& Value)
	{
		size_t iDivider;
		string	str(lpsz);
		bool Negative = false;

		/* Hours:Minutes:Seconds field */

		while (*lpsz == ' ' || *lpsz == '\t') lpsz++;
		if (*lpsz == '-') Negative = true;

		Int64	nA;
		if (!Int64_TryParse(str.c_str(), wb::NumberStyles::Integer, nA)) return false;
		nA = abs(nA);			// Will be accounted for by "Negative" flip, which also handles the case of 0 hours but still negative.

		iDivider = str.find(':');
		if (iDivider == string::npos) {

			Value.m_nElapsedSeconds = nA;
			if (Negative) Value.m_nElapsedSeconds = -Value.m_nElapsedSeconds;

			iDivider = str.find('.');
			if (iDivider == string::npos) { Value.m_nElapsedNanoseconds = 0; return true; }

			str = str.substr(iDivider);
			double nano;
			if (!Double_TryParse(str.c_str(), wb::NumberStyles::Float, nano)) return false;

			Value.m_nElapsedNanoseconds = (Int32)(nano / time_constants::g_dSecondsPerNanosecond);
			if (Value.m_nElapsedSeconds < 0) Value.m_nElapsedNanoseconds = -Value.m_nElapsedNanoseconds;
			return true;
		}

		/* Minutes:Seconds field */

		str = str.substr(iDivider + 1);

		Int64	nB;
		if (!Int64_TryParse(str.c_str(), wb::NumberStyles::Integer, nB)) return false;

		iDivider = str.find(':');
		if (iDivider == string::npos) {

			Value.m_nElapsedSeconds = (nA * time_constants::g_nSecondsPerMinute) + nB;
			if (Negative) Value.m_nElapsedSeconds = -Value.m_nElapsedSeconds;

			iDivider = str.find('.');
			if (iDivider == string::npos) { Value.m_nElapsedNanoseconds = 0; return true; }

			str = str.substr(iDivider);
			double nano;
			if (!Double_TryParse(str.c_str(), wb::NumberStyles::Float, nano)) return false;

			Value.m_nElapsedNanoseconds = (Int32)(nano / time_constants::g_dSecondsPerNanosecond);
			if (Value.m_nElapsedSeconds < 0) Value.m_nElapsedNanoseconds = -Value.m_nElapsedNanoseconds;
			return true;
		}

		/* Seconds field */

		str = str.substr(iDivider + 1);

		Int64	nC;
		if (!Int64_TryParse(str.c_str(), wb::NumberStyles::Integer, nC)) return false;
		Value.m_nElapsedSeconds = (nA * time_constants::g_nSecondsPerHour) + (nB * time_constants::g_nSecondsPerMinute) + nC;
		if (Negative) Value.m_nElapsedSeconds = -Value.m_nElapsedSeconds;

		iDivider = str.find('.');
		if (iDivider == string::npos) { Value.m_nElapsedNanoseconds = 0; return true; }

		str = str.substr(iDivider);
		double nano;
		if (!Double_TryParse(str.c_str(), wb::NumberStyles::Float, nano)) return false;

		Value.m_nElapsedNanoseconds = (Int32)(nano / time_constants::g_dSecondsPerNanosecond);
		if (Value.m_nElapsedSeconds < 0) Value.m_nElapsedNanoseconds = -Value.m_nElapsedNanoseconds;
		return true;
	}

	inline /*static*/ TimeSpan TimeSpan::Parse(const char* psz)
	{
		TimeSpan ret;
		if (!TryParse(psz, ret))
			throw FormatException(S("Unable to parse time span."));
		return ret;
	}

		/** String conversions **/

#ifndef _MSC_VER
#define sprintf_s sprintf
#endif

	inline string TimeSpan::ToString(int Precision /*= 9*/) const
	{
		Int64 nDays;
		Int32 nHours, nMinutes, nSeconds, nNanoseconds;
		Get(nDays, nHours, nMinutes, nSeconds, nNanoseconds);
		// Note: Get(...) returns all positive values.  Must call IsNegative() to
		// find negative cases.

		if (Precision < 1)
		{
			if (nNanoseconds > 500000000)
			{
				nSeconds++;
				if (nSeconds >= 60)
				{
					nSeconds = 0; nMinutes++;
					if (nMinutes >= 60)
					{
						nMinutes = 0; nHours++;
						if (nHours >= 24)
						{
							nHours = 0; nDays++;
						}
					}
				}
			}
			else if (nNanoseconds < -500000000)
			{
				nSeconds--;
				if (nSeconds <= -60)
				{
					nSeconds = 0; nMinutes--;
					if (nMinutes <= -60)
					{
						nMinutes = 0; nHours--;
						if (nHours <= -24)
						{
							nHours = 0; nDays--;
						}
					}
				}
			}

			char tmp[64];
			if (nHours < 24)
				sprintf_s(tmp, S("%d:%02d:%02d hours"), nHours, nMinutes, nSeconds);
			else
				sprintf_s(tmp, S("%lld days %d:%02d:%02d hours"), nDays, nHours, nMinutes, nSeconds);
			if (IsNegative()) return string(S("-")) + tmp;
			else return tmp;
		}
		else
		{
			double dSeconds = (double)nSeconds + (double)nNanoseconds * time_constants::g_dSecondsPerNanosecond;

			char tmp[64];
			if (abs(nHours) < 24)
				sprintf_s(tmp, S("%d:%02d:%02.*f hours"), nHours, nMinutes, Precision, dSeconds);
			else
				sprintf_s(tmp, S("%lld days %d:%02d:%02.*f hours"), nDays, nHours, nMinutes, Precision, dSeconds);
			if (IsNegative()) return string(S("-")) + tmp;
			else return tmp;
		}

	}// End of ToString()			

	inline string TimeSpan::ToShortString() const
	{
		// Returns string as "XX Days", showing only highest-level unit.

		/** Attention must be paid to the rounding that occurs in different
			Get...() calls. **/

		string Sign = S("");
		if (IsNegative()) Sign = S("-");

		char tmp[64];
		Int64 nTotalDays = abs(GetTotalDays());

		if (nTotalDays > 90ll)
		{
			if (GetApproxTotalYears() >= 2ll)			// Note: We allow months up to 24...
			{
				sprintf_s(tmp, S("%lld years"), GetApproxTotalYears());
				return Sign + tmp;
			}
			else
			{
				sprintf_s(tmp, S("%lld months"), GetApproxTotalMonths());
				return Sign + tmp;
			}
		}
		else	// Else( less than 90 days )
		{
			if (nTotalDays >= 2ll)					// Note: We allow hours up to 48...
			{
				sprintf_s(tmp, S("%lld days"), GetTotalDays());
				return Sign + tmp;
			}
			else if (abs(GetTotalMinutes()) > 90)			// Note: We allow minutes up to 90...
			{
				sprintf_s(tmp, S("%lld hours"), GetTotalHours());
				return Sign + tmp;
			}
			else
			{
				if (abs(GetTotalSeconds()) > 90) {					// Note: We allow seconds up to 90...

					sprintf_s(tmp, S("%lld minutes"), llabs(GetTotalMinutes()));
					return Sign + tmp;
				}

				Int64 TotalSeconds = abs(GetTotalSeconds());
				if (TotalSeconds > 30) {
					sprintf_s(tmp, S("%lld seconds"), llabs(GetTotalSeconds()));
					return Sign + tmp;
				}

				double dSeconds = (double)TotalSeconds + (double)GetNanoseconds() * time_constants::g_dSecondsPerNanosecond;

				if (TotalSeconds > 5) {
					sprintf_s(tmp, S("%.1f seconds"), dSeconds);
					return Sign + tmp;
				}

				if (dSeconds > 0.030) {
					sprintf_s(tmp, S("%.3f seconds"), dSeconds);
					return Sign + tmp;
				}

				if (dSeconds >= 0.000030) {
					sprintf_s(tmp, S("%.6f seconds"), dSeconds);
					return Sign + tmp;
				}

				sprintf_s(tmp, S("%d nanoseconds"), GetNanoseconds());
				return Sign + tmp;
			}
		}
	}// End of ToShortString()

#ifndef _MSC_VER
#undef sprintf_s
#endif
}

#endif	// __TimeSpan_h__

//	End of TimeSpan.h


