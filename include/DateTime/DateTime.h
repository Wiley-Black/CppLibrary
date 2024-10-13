/**	DateTime.h
	Copyright (C) 2014-2021 by Wiley Black (TheWiley@gmail.com)
**/

/*
	A class for representing dates and times.

	- Signed 128-bit integer implementation supports any time with nanosecond precision.  
	- Time zone representation.  Time always stored internally as UTC, but a bias is represented which
		allows support for any specified time zone.
	- The data content is comprised of:
		- Number of seconds elapsed since epoch (signed 64-bit).
		- Number of nanoseconds elapsed from second (unsigned 32-bit).
		- Bias (in seconds) from UTC (negative west of the prime meridian, positive east)
	- The epoch is year zero, time zero.

	Conversions:

		To/From UTC/local-time-zone:

			// Note the extra zero on the end of the following contructor call (bias parameter.)
		DateTime	dtUTC( 1980, 1, 21, 0, 0, 0, 0 );			// Jan 21st, 1980 at midnight in UTC time.
		DateTime	dtAsLocalTime = dtUTC.asLocalTime();		// Now in local time.
		DateTime	dtBackAgain = dtAsLocalTime.asUTC();		// Jan 21st, 1980 at midnight in UTC time.
		assert( dtUTC == dtBackAgain );

		To/From an MFC CTime:

		DateTime	dtJan21_1980( 1980, 1, 21, 0, 0, 0 );		// Jan 21st, 1980 at midnight in local time zone.
		time_t		timeValue	= (time_t)dtJan21_1980;			// Uses the time_t conversion.  Now in UTC.
		CTime		tmJan21_1980( dtJan21_1980 );				// Uses the time_t conversion.  CTime in local time.
		assert( timeValue == tmJan21_1980.GetTime() );			// CTime's internal representation is the time_t.
		DateTime	dtBackAgain( tmJan21_1980 );				// Uses the CTime constructor.  Remains local time.
		assert( dtBackAgain == dtJan21_1980 );

		To/From a time_t:
			* See known issues.

		DateTime	dtJan21_1980( 1980, 1, 21, 0, 0, 0 );		// Jan 21st, 1980 at midnight in local time zone.
		time_t		timeJan21_1980_UTC	= (time_t)dtJan21_1980;	// Uses the time_t conversion.  Becomes UTC time.
		DateTime	dtJan21_1980_UTC( timeJan21_1980_UTC );		// Now in UTC time.

	Implementation:

		Platform requires support for 64-bit signed integers.

		Represents date/time as a 64-bit signed integer as the number of seconds
		elapsed since midnight (00:00:00), January 1, 0.  This counts year zero
		as a year.  Negative numbers of greater magnitude (i.e. -500 vs -5) are
		earlier in a timeline (i.e. -31M would be 1 B.C. and -63M would be 2 B.C.)

		Gregorian Calendar Rules Applied:
		-	If the year is divisible by 4 but does not end in 00, then the year is a leap year, with 366 days. 
			Examples: 1996, 2004. 
		-	If the year is not divisible by 4, then the year is a nonleap year, with 365 days. 
			Examples: 2001, 2002, 2003, 2005. 
		-	If the year ends in 00 but is not divisible by 400, then the year is a nonleap year, with 365 days. 
			Examples: 1900, 2100. 
		-	If the year ends in 00 and is divisible by 400, then the year is a leap year, with 366 days. 
			Examples: 1600, 2000, 2400. 
		-	The year zero was a leap year.
		-	Months:			Jan, Mar, May, Jul, Aug, Nov, Dec have 31 days.
		-	Months:			        Apr, Jun,      Sep, Oct   have 30 days.
		-	Non-Leap Year:  Feb has 28 days.  Year has 365 days.
		-	Leap Year:		Feb has 29 days.  Year has 366 days.		

	Regarding Time Zones:

		The Win32 GetTimeZoneInformation() API defines all translations between UTC time and local time based on the following formula:
		
			UTC = local time + bias
		
		By contrast, ISO-8601 specifies the civil time and describes the offset to UTC to get the civil time.  For example:
		
			If UTC is:           2024-05-21 15:30:01Z
			EST (UTC-05) is:     2024-05-21 10:30:01-05:00

		The DateTime class defines the bias as the bias of the represented time from UTC.  Therefore, it
		is opposite the GetTimeZoneInformation() API definition above and follows the relation:
		
			UTC + bias = local time

			UTC (in seconds) = m_nSeconds - m_nBias.

		The above EST time example would have a bias value of (-5 * 60) because that is the represented time's bias from UTC.

	Limitations:

		Projection of times into different eras is not performed.  For example, calculating a time value
		in an era when daylight savings time was not applied will provide the time on today's Gregorian
		calendar with today's daylight svaings rules, but will not correct for differences from the 
		era.  Some examples can be identified:

		- Eras with different daylight savings time rules may result in inaccurate time zone biases.
		- Eras using different calendars.  For example, a date prior to 1582 would have been specified
		  on the Julian calendar, however the adjustment will not be applied by DateTime automatically.
		- Calculating a time with a bias set by daylight savings time on (off) and then adding or subtracting
		  a change in time to a period when daylight savings time would be off (on).  The original bias
		  will be retained.

		The bias for "local times" is calculated with respect to the current system bias, not the bias that
		would apply at a different date.
*/
/////////

#ifndef __DateTime_h__
#define __DateTime_h__

/////////
//	Dependencies
//

#if (defined(_MSC_VER) && !defined(_INC_WINDOWS))
	#error Include Windows.h before this header.
#endif

#include "../Platforms/Platforms.h"
#include "../Foundation/Exceptions.h"
#include "../Foundation/STL/Text/String.h"
#include "TimeConstants.h"

#if (defined(_MSC_VER))
#include <sys\types.h>
#include <stdlib.h>
#include <assert.h>
#include <float.h>
#include <time.h>
#endif

#ifndef _WINDOWS
typedef struct _FILETIME {
  UInt32 dwLowDateTime;
  UInt32 dwHighDateTime;
} FILETIME, *PFILETIME;

typedef struct _SYSTEMTIME {
	WORD wYear;
	WORD wMonth;
	WORD wDayOfWeek;
	WORD wDay;
	WORD wHour;
	WORD wMinute;
	WORD wSecond;
	WORD wMilliseconds;
} SYSTEMTIME, * PSYSTEMTIME, * LPSYSTEMTIME;
#endif

namespace wb
{
	class TimeSpan;

	class DateTime
	{
	protected:

		Int64	m_nSeconds;				// Time, in seconds, since epoch (Zero).
		UInt32	m_nNanoseconds;			// Nanoseconds after m_nSeconds.
		Int32	m_nBias;				// The bias (in seconds) from UTC.  UTC (in seconds) = m_nSeconds - m_nBias.

			// The Get...Remainder() functions always operate on positive remainders.  This makes 
			// sense because only years can be negative.  At the 'year-to-month' remainder
			// transition, negative years cause the remainder to be subtracted out of 1 year.

			// GetYearAndRemainder() uses remainder as an output only.
			// For years B.C., the return year will be negative but the remainder
			// will be positive (absolute value).
		int		GetYearAndRemainder( UInt32& nRemainder ) const;

			// GetMonthFromRemainder() uses remainder as both an input and an output.
			// For years B.C., the remainder returned from GetMonthFromRemainder() will have
			// been subtracted from 1 year.
		int		GetMonthFromRemainder( UInt32& nRemainder, int nYear, bool bLeapYear ) const;

			// GetDaysFromRemainder() uses remainder as both an input and an output.
			// It returns the number of WHOLE days contained in the input remainder.
			// For day-of-month presentation, add one to the return value.
		int		GetDaysFromRemainder( UInt32& nRemainder ) const;

			// These Get...FromRemainder() uses remainder as both an input and an output.
			// It returns (and removes) the number of WHOLE units contained in the remainder.
			// The remainder of the call to GetMinutesFromRemainder() is the number of seconds.
		int		GetHoursFromRemainder( UInt32& nRemainder ) const;
		int		GetMinutesFromRemainder( UInt32& nRemainder ) const;

	public:

		enum { 
			UTC				= 0,
			LocalTimeZone	= Int32_MaxValue
		};

		enum {
			January	= 1,
			February,
			March,
			April,
			May,
			June,
			July,
			August,
			September,
			October,
			November,
			December
		};

		enum DayOfWeek {
			Sunday = 0,
			Monday,
			Tuesday,
			Wednesday,
			Thursday,
			Friday,
			Saturday
		};

			/** For example, for January returns 31.  
				The last day of January is always January 31st. **/
		static int GetDaysInMonth( int nMonth, bool bLeapYear );
		static int GetDaysInMonth( int nMonth, int nYear );

			/** Day specified as (1-31) **/
			/** Month specified as (1-12) **/
			/** Year specified as (-200 billion to +200 billion) **/
			/** Hour specified as (0-23) **/
			/** Minutes and seconds specified as (0-59) **/
			/** Bias is the bias from UTC time, in minutes (+/- 1439 or the special value 'LocalTimeZone') **/
			/** UTC = time/date-specified - Bias **/
			/** e.g. For a time/date specified in U.S. Mountain Standard Time, outside DST, the Bias would be -(7 * 60). **/				
		DateTime(int nYear, int nMonth, int nDay, int nHour, int nMinute, int nSecond, int nNanosecond = 0, int nBiasMinutes = LocalTimeZone);
		DateTime(const DateTime&);
		DateTime(DateTime&&);
		DateTime(FILETIME);
		DateTime(const SYSTEMTIME&, int nBiasMinutes = LocalTimeZone);
		DateTime(time_t);				// Note: The value will be in UTC time.  Conversions are possible, see usage.
		DateTime();	

		int			GetYear() const;
		int			GetMonth() const;
		int			GetDay() const;
		int			GetHour() const;		// Returns the hour as (0-23)	
		int			GetMinute() const;
		int			GetSecond() const;

		int			GetHourIn12HourFormat() const;
		bool		IsAM() const;
		bool		IsPM() const;

		void		GetDayOfWeek( DayOfWeek& nValue ) const;
		string		GetDayOfWeek() const;

		string		GetMonthAsString() const;

			/**	Now() returns a DateTime object representing the date/time at the time of the function call in local time.
				UtcNow() returns a DateTime object representing the date/time at the time of the function call in UTC.				
			**/
		static DateTime		Now();
		static DateTime		UtcNow();
		// static DateTime		Zero();			// Returns the date time corresponding to midnight, January 1, 0000 UTC.

			/** GetCurrentDate()
				Returns a CDateTime object corresponding to midnight of the current day in UTC.
			**/
		static DateTime		GetCurrentDate();

			/** SetSystemTime()
				Sets the current system time to match the DateTime object's value. **/
		static void			SetSystemTime(const DateTime& to);

			// The following Get() calls are much less computationally expensive than individual Get...() calls.
		void	Get(int& nYear, int& nMonth, int& nDay, int& nHour, int& nMinute, int& nSecond, int& nNanosecond) const;
		void	Get(int& nYear, int& nMonth, int& nDay, int& nHour, int& nMinute, int& nSecond) const;
		void	GetDate(int& nYear, int& nMonth, int& nDay) const;
		void	GetTime(int& nHours, int& nMinutes, double& dSeconds) const;
		void	GetTime(int& nHours, int& nMinutes, int& nSeconds) const;

		bool	IsLeapYear() const;							// Returns true if the current year is a leap year.
		static bool IsLeapYear(int nYear);				// Returns true if 'nYear' is a leap year.

			/** GetSecondsIntoYear()
				GetSecondsIntoYear() returns the number of true seconds elapsed from the beginning of the year.
				Use this (probably after converting to UTC) in combination with GetYear() as one method of
				disassembling a DateTime value.
			**/
		UInt32	GetSecondsIntoYear() const;		

			// For range of parameters, see matching constructor.		
		void	Set(int nYear, int nMonth, int nDay, int nHour, int nMinute, int nSecond, int nNanoseconds = 0, int nBiasMinutes = LocalTimeZone);
		void	Set(time_t);
		void	Set(UInt64 Seconds, UInt32 Nanoseconds, int nBiasSeconds);

			// The Add...() functions can accept negative values.
			// The Add...() functions can accept values beyond one unit (for example, you can add 3600 seconds.)
			// Effects such as leap-years are automatically accounted for.
		void	AddTime( int nHours, int nMinutes, double dSeconds );	// Adds up to 24 hours to the value of this object.
		void	AddTime( int nHours, int nMinutes, int nSeconds );		// Adds up to 24 hours to the value of this object.
		void	AddDays( int nDays = 1 );		// Adds N days (24-hours ea) to the value of this object.
		void	AddMonths( int nMonths = 1 );	// Adds N month (28,29,30, or 31 days ea) to the value of this object.
		void	AddYears( int nYears = 1 );		// Adds N year (365 or 366 days ea) to the value of this object.
		void	Add( int nYears, int nMonths, int nDays, int nHours, int nMinutes, double dSeconds );
		void	Add( int nYears, int nMonths, int nDays, int nHours, int nMinutes, int nSeconds );	

			/** Operations **/
	
		bool operator==( const DateTime& ) const;
		bool operator!=( const DateTime& ) const;
		bool operator<=( const DateTime& ) const;
		bool operator<( const DateTime& ) const;
		bool operator>=( const DateTime& ) const;
		bool operator>( const DateTime& ) const;

		const DateTime& operator=( const DateTime& );
		const DateTime& operator+=( const TimeSpan& timeSpan );
		const DateTime& operator-=( const TimeSpan& timeSpan );
		DateTime operator+( const TimeSpan& tmSpan ) const;
		DateTime operator-( const TimeSpan& tmSpan ) const;
		TimeSpan operator-( const DateTime& ) const;			

			/** Conversions **/

		operator time_t() const;				// Automatically returns UTC time based on the given bias.
		operator tm() const;					// Returns local time zone time.
		operator bool() const;					// Returns true if time is initialized.	
		bool operator!() const;	

		DateTime	asUTC() const;				// Converts the time/date value to UTC.
		DateTime	asLocalTime() const;		// Converts the time/date value to the local time zone.

		bool    IsUTC() const { return m_nBias == 0; }  // Returns true if the object is an UTC representation

		string Format( const char *lpszFormatStr ) const;
		string asLongString() const;											// Returns string as "HH:MM:SS x.m. on Weekday, Month XX, YYYY", 12-hr clock
		string asPresentationString( bool bSeconds = true ) const;				// Returns string as "HH:MM:SS x.m. Weekday, Month XX, YYYY", 12-hr clock
		string asMediumString( bool bSeconds = true, bool b24Hr = true, bool bYear = true ) const;		// Returns string as "Wkd Mon XX[ YYYY] HH:MM[:SS][ XM]"
		string asShortString() const;											// Returns string as "Mon XX YY HH:MM:SS", 24-hr clock
		string asNumericString() const;											// Returns string as "DD.MM.YYYY HH:MM:SS", 24-hr clock
		string asMilString() const;												// Returns string as "DDMonYY HH:MM:SS", 24-hr clock
		string asDateString( bool bYear = true ) const;							// Returns string as "Weekday, Month XX[, YYYY]"
		string asTimeString( bool bSeconds = true, bool b24Hr = true ) const;	// Returns string as "HH:MM[:SS]" or "HH:MM[:SS] XM"
		string asDebugString() const;											// Returns string as "DD.MM.YYYY HH.MM.SS", 24-hr clock
		string asInternetString() const;										// Returns string as "Wkd, DD Mnt YYYY HH:MM:SS GMT", 24-hr clock (see RFC 822 & 1123)	
		string asISO8601(int Precision = 6) const;								// Returns string as "YYYY-MM-DDTHH:MM:SS.ssssss[+/-HH:MM or Z]" as in ISO 8601.
		void asMSDOS(UInt16& date, UInt16& time) const;

		string		ToString() const;							// Same as 'asISO8601()'.  Recommended format for storage/transmission.
		static bool	TryParse(const char*, DateTime&);			// Reads the 'asInternetString()' (always UTC) or 'asPresentationString()' (assumes local time) formats.
		static DateTime Parse(const char*);
		static DateTime FromMSDOS(UInt16 date, UInt16 time);

		FILETIME ToFILETIME() const;
		SYSTEMTIME ToSYSTEMTIME() const;

			// Returns the number of seconds since Midnight (00:00:00) on Year zero.
		Int64	GetSeconds() const { return m_nSeconds; }

			// Returns the number of nanoseconds after the "seconds" value.
		Int32	GetNanoseconds() const { return m_nNanoseconds; }

			// Returns the number of seconds difference between UTC time and the value.  UTC = Value - Bias.
			// This is specified with the commonly used time zone convention.  For example, Arizona would have
			// a m_nBias value of -7h (-25200 seconds).
		Int32	GetBias() const { return m_nBias; }

			// Returns the number of seconds since Midnight (00:00:00) on Year zero after converting to UTC time.
		Int64	GetUTCSeconds() const { return m_nSeconds - m_nBias; }

			// Returns the number of seconds since Midnight (00:00:00) on Year zero after converting to local time.		
			// UTC = LocalValue - Bias -> LocalValue = UTC + Bias.
		Int64	GetLocalSeconds() const { return (m_nSeconds - (Int64)m_nBias + (Int64)GetLocalBias()); }

			// Returns the bias, in seconds, applied when working in the local time zone.  UTC = Value - Bias.
		static int	GetLocalBias();

		/* Syntactically nice, but being static variables makes it harder on "header-only" library. 
		static DateTime Minimum;
		static DateTime Maximum;
		static DateTime Zero;
		*/

		static DateTime GetMinimumValue();			// Retrieves the lowest allowed DateTime value.
		static DateTime GetMaximumValue();			// Retrieves the highest allowed DateTime value.
		static DateTime GetZero();					// Retrieves a zero DateTime value.  Often better than using "Minimum" to avoid rollovers.
	};

	#define DateTime_MinValue (DateTime::Minimum)
	#define DateTime_MaxValue (DateTime::Maximum)

	/** Non-class operations **/

	inline string to_string(const DateTime& Value) { return Value.ToString(); }
}

//	Late dependency
#include "TimeSpan.h"

namespace wb
{
	/////////
	//	DateTime Constants
	//

	/** Relative offset for the time_t type (both are in units of seconds but at different initial times.)
		Definition of the time_t type in Win32:
		The number of seconds elapsed since midnight (00:00:00), January 1, 1970, coordinated universal time.

		Years divisible by 4:	+	492. (Not including year zero.)
		Years ending in 00:		-	19.	(Not including year zero.)
		Years divisible by 400:	+	4. (Not including year zero.)
		Year zero:				+	1 (A leap year.)
								--------
		Total Leap Years:			478. (Including year zero.)
		Total Non-Leap Years:	   1492.
	**/	
	// Formerly:	DateTime::g_nOffsetForTimeT
#define DateTime_g_nOffsetForTimeT		\
	(const Int64) ((time_constants::g_nSecondsPerLeapYear * 478ll) + (time_constants::g_nSecondsPerNonLeapYear * 1492ll))

	/** Relative offset for the FILETIME type
		Definition of the FILETIME type in Win32:
		The number of 100-nanosecond intervals since January 1, 1601.
		The value will be converted to the number of seconds elapsed since January 1, 1601 before offseting.

		Years divisible by 4:	+	400. (Not including year zero.)
		Years ending in 00:		-	16.	(Not including year zero.)
		Years divisible by 400:	+	4. (Not including year zero.)
		Year zero:				+	1 (A leap year.)
								--------
		Total Leap Years:			389. (Including year zero.)
		Total Non-Leap Years:	   1212.
	**/
	// Formerly:  DateTime::g_nOffsetForFiletime
#define DateTime_g_nOffsetForFiletime	\
	(const Int64) ((time_constants::g_nSecondsPerLeapYear * 389ll) + (time_constants::g_nSecondsPerNonLeapYear * 1212ll))

#if 0	// Synactically nice, but complicates "header-only" library.
	/*static*/ DateTime DateTime::Minimum = DateTime::GetMinimumValue();
	/*static*/ DateTime DateTime::Maximum = DateTime::GetMaximumValue();
	/*static*/ DateTime DateTime::Zero = DateTime::GetZero();
#endif

	/*static*/ inline DateTime DateTime::GetMinimumValue() { DateTime ret; ret.m_nSeconds = Int64_MinValue; ret.m_nNanoseconds = 0; ret.m_nBias = 0; return ret; }
	/*static*/ inline DateTime DateTime::GetMaximumValue() { DateTime ret; ret.m_nSeconds = Int64_MaxValue; ret.m_nNanoseconds = 0; ret.m_nBias = 0; return ret; }
	/*static*/ inline DateTime DateTime::GetZero() { return DateTime(0, 1, 1, 0, 0, 0, DateTime::UTC); }

	/////////
	//	Inline functions
	//	

	inline DateTime::DateTime(){ m_nSeconds = 0ull; m_nNanoseconds = 0; m_nBias = 0; }
	inline DateTime::DateTime(const DateTime& cp){ m_nSeconds = cp.m_nSeconds; m_nNanoseconds = cp.m_nNanoseconds; m_nBias = cp.m_nBias; }
	inline DateTime::DateTime(DateTime&& cp){ m_nSeconds = cp.m_nSeconds; m_nNanoseconds = cp.m_nNanoseconds; m_nBias = cp.m_nBias; }
	inline DateTime::DateTime(time_t nValue){ Set(nValue); }
	inline DateTime::DateTime(int nYear, int nMonth, int nDay, int nHour, int nMinute, int nSecond, int nNanosecond /*= 0*/, int nBiasMinutes /*= LocalTimeZone*/)
	{
		Set(nYear, nMonth, nDay, nHour, nMinute, nSecond, nNanosecond, nBiasMinutes);
	}	
	inline DateTime::DateTime(FILETIME ft)
	{
		UInt64	iFiletime = ((UInt64)ft.dwLowDateTime) | ((UInt64)ft.dwHighDateTime << 32);
			// in units of 0.0000001 seconds (100 nanosecond units)
			// 50000000 in units of 0.0000001 seconds = 5.0 seconds, so /10000000 = 5 seconds
		static const UInt64 PerSecond = 10000000ull;
		m_nSeconds = (iFiletime / PerSecond) + DateTime_g_nOffsetForFiletime;
		m_nNanoseconds = iFiletime % PerSecond;
		m_nBias	= 0;
	}
	inline FILETIME DateTime::ToFILETIME() const
	{
		// Note: FILETIME is only valid from year 1601 to 30827.		
		static const UInt64 PerSecond = 10000000ull;
		DateTime UTC = asUTC();
		UInt64	iFiletime = UTC.m_nSeconds - DateTime_g_nOffsetForFiletime;
		iFiletime *= PerSecond;
		iFiletime += UTC.m_nNanoseconds / 100ull;
		FILETIME ret;
		ret.dwLowDateTime = (UInt32)(iFiletime);
		ret.dwHighDateTime = (UInt32)(iFiletime >> 32ull);
		return ret;
	}
	inline DateTime::DateTime(const SYSTEMTIME& st, int nBiasMinutes /*= LocalTimeZone*/)
	{
		Set(st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds * 1000000, nBiasMinutes);
	}
	inline SYSTEMTIME DateTime::ToSYSTEMTIME() const
	{
		int nYear, nMonth, nDay, nHour, nMinute, nSecond, nNanosecond;
		Get(nYear, nMonth, nDay, nHour, nMinute, nSecond, nNanosecond);
		if (nYear < 0 || nYear >(Int32)UInt16_MaxValue) throw NotSupportedException(S("Cannot calculate SYSTEMTIME for years outside of 16-bit range."));
		DayOfWeek dow;
		GetDayOfWeek(dow);
		SYSTEMTIME st;
		st.wYear = nYear;
		st.wMonth = nMonth;		
		st.wDayOfWeek = (int)dow;	// enum uses same numbering values as SYSTEMTIME
		st.wDay = nDay;
		st.wHour = nHour;
		st.wMinute = nMinute;
		st.wSecond = nSecond;
		st.wMilliseconds = nNanosecond / 1000000;
		return st;
	}

	inline /*static*/ bool DateTime::IsLeapYear(int nYear)
	{
			/** Determine if current year is a leap-year **/

			/** Test Cases
				----------
				year & 3			Year 0 & 3 = 0 (true)
				(A)					Year 99 & 3 = 3 (false)
									Year 100 & 3 = 0 (true)
									Year 2000 & 3 = 0 (true)

				year % 100			Year 0 % 100 = 0 (false)
				(B)					Year 99 % 100 = 99 (true)
									Year 100 % 100 = 0 (false)
									Year 2000 % 100 = 0 (false)

				!(year % 400)		Year 0 % 400 = 0 (true)
				(C)					Year 99 % 400 = 99 (false)
									Year 100 % 400 = 100 (false)
									Year 2000 % 400 = 0 (true)

				A && (B || C)		Year 0: A=true, B=false, C=true ==> true (Leap-Year)
									Year 99: A=false, B=true, C=false ==> false (Non-Leap-Year)
									Year 100: A=true, B=false, C=false ==> false (Non-Leap-Year)
									Year 2000: A=true, B=false, C=true ==> true (Leap-Year)
			**/

		return ((abs(nYear) & 3) == 0) && (((abs(nYear) % 100) != 0) || (((abs(nYear) % 400) == 0)));
	}

	inline bool	DateTime::IsLeapYear() const { return IsLeapYear( GetYear() ); }

	inline int DateTime::GetDaysFromRemainder( UInt32& nRemainder ) const {
		int nDays = nRemainder / time_constants::g_n32SecondsPerDay;
		nRemainder -= nDays * time_constants::g_n32SecondsPerDay;
		return nDays;
	}

	inline int DateTime::GetHoursFromRemainder( UInt32& nRemainder ) const {
		int nHours = nRemainder / time_constants::g_n32SecondsPerHour;
		nRemainder -= nHours * time_constants::g_n32SecondsPerHour;
		return nHours;
	}

	inline int DateTime::GetMinutesFromRemainder( UInt32& nRemainder ) const {
		int nMinutes = nRemainder / 60;
		nRemainder -= nMinutes * 60;
		return nMinutes;
	}

	inline int DateTime::GetYear() const { UInt32 nRemainder; return GetYearAndRemainder(nRemainder); }
	inline int DateTime::GetMonth() const { 
		UInt32 nRemainder; int nYear = GetYearAndRemainder(nRemainder); 
		return GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
	}

	inline int DateTime::GetDay() const { 
		UInt32 nRemainder; int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		return GetDaysFromRemainder( nRemainder ) + 1;
	}

	inline int DateTime::GetHour() const { 
		UInt32 nRemainder; int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		GetDaysFromRemainder( nRemainder );
		return GetHoursFromRemainder( nRemainder );
	}

	inline int DateTime::GetMinute() const { 
		UInt32 nRemainder; int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		GetDaysFromRemainder( nRemainder );
		GetHoursFromRemainder( nRemainder );
		return GetMinutesFromRemainder( nRemainder );
	}

	inline int DateTime::GetSecond() const { 
		UInt32 nRemainder; int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		GetDaysFromRemainder( nRemainder );
		GetHoursFromRemainder( nRemainder );
		GetMinutesFromRemainder( nRemainder );
		return (int)nRemainder;
	}

	inline int DateTime::GetHourIn12HourFormat() const { 
		int Hour = GetHour(); 
		if (Hour == 0) return 12;
		if (Hour == 12) return 12;
		if (Hour >= 12) Hour -= 12;
		return Hour;
	}

	inline bool	DateTime::IsAM() const {
		int Hour = GetHour();
		return (Hour < 12);
	}

	inline bool	DateTime::IsPM() const {
		int Hour = GetHour();
		return (Hour >= 12);
	}

	inline void DateTime::Get(int& nYear, int& nMonth, int& nDay, int& nHour, int& nMinute, int& nSecond, int& nNanosecond) const {
		UInt32 nRemainder; 
		nYear = GetYearAndRemainder(nRemainder);
		nMonth = GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		nDay = GetDaysFromRemainder( nRemainder ) + 1;
		nHour = GetHoursFromRemainder( nRemainder );
		nMinute = GetMinutesFromRemainder( nRemainder );
		nSecond = (int)nRemainder;
		nNanosecond = m_nNanoseconds;
	}

	inline void DateTime::Get(int& nYear, int& nMonth, int& nDay, int& nHour, int& nMinute, int& nSecond) const {
		UInt32 nRemainder; 
		nYear = GetYearAndRemainder(nRemainder);
		nMonth = GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		nDay = GetDaysFromRemainder( nRemainder ) + 1;
		nHour = GetHoursFromRemainder( nRemainder );
		nMinute = GetMinutesFromRemainder( nRemainder );
		nSecond = (int)nRemainder;
	}

	inline void DateTime::GetDate(int& nYear, int& nMonth, int& nDay) const {
		UInt32 nRemainder; 
		nYear = GetYearAndRemainder(nRemainder);
		nMonth = GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		nDay = GetDaysFromRemainder( nRemainder ) + 1;
	}

	inline void	DateTime::GetTime(int& nHours, int& nMinutes, double& dSeconds) const {
		UInt32 nRemainder; 
		int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		GetDaysFromRemainder( nRemainder );
		nHours = GetHoursFromRemainder( nRemainder );
		nMinutes = GetMinutesFromRemainder( nRemainder );
		dSeconds = (int)nRemainder;
		dSeconds += (double)m_nNanoseconds * time_constants::g_dSecondsPerNanosecond;
	}

	inline void	DateTime::GetTime(int& nHours, int& nMinutes, int& nSeconds) const {
		UInt32 nRemainder; 
		int nYear = GetYearAndRemainder(nRemainder);
		GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) );
		GetDaysFromRemainder( nRemainder );
		nHours = GetHoursFromRemainder( nRemainder );
		nMinutes = GetMinutesFromRemainder( nRemainder );
		nSeconds = (int)nRemainder;
	}

	inline void DateTime::GetDayOfWeek(DayOfWeek& dow) const
	{
		if( m_nSeconds <= time_constants::g_n32SecondsPerDay )
		{
			int nDays = (int)(1 - (m_nSeconds / time_constants::g_n32SecondsPerDay)) - 1;
			dow = (DayOfWeek)(6 - (nDays % 7));
		}
		else
		{
				/* 1. Calculate # of days since Midnight, January 1st, Year 0 */
			int		nDays	= (int)(m_nSeconds / time_constants::g_n32SecondsPerDay) - 1;
			dow = (DayOfWeek)(nDays % 7);
		}

		assert( ((int)dow) >= 0 && ((int)dow) <= 6 );
	}

	inline string DateTime::GetDayOfWeek() const
	{
			/* 1. Calculate # of days since Midnight, January 1st, Year 0 */
		int		nDays	= (int)(m_nSeconds / time_constants::g_n32SecondsPerDay) - 1;
			/* Jan 1, 0 was a Saturday it would seem.  (Since the Gregorian calender
				doesn't actually extend back that far without corrections, this is 
				only partially true.) */
		switch( nDays % 7 ){ 
		default:	assert(1);
		case Sunday: return string(S("Sunday"));
		case Monday: return string(S("Monday"));
		case Tuesday: return string(S("Tuesday"));
		case Wednesday: return string(S("Wednesday"));
		case Thursday: return string(S("Thursday"));
		case Friday: return string(S("Friday"));
		case Saturday: return string(S("Saturday"));
		}
	}

	inline string DateTime::GetMonthAsString() const {
		switch( GetMonth() )
		{
		case January:	return S("January");
		case February:	return S("February");
		case March:		return S("March");
		case April:		return S("April");
		case May:		return S("May");
		case June:		return S("June");
		case July:		return S("July");
		case August:	return S("August");
		case September:	return S("September");
		case October:	return S("October");
		case November:	return S("November");
		case December:	return S("December");
		default:		return S("Undefined");
		}
	}

	inline UInt32 DateTime::GetSecondsIntoYear() const {
		UInt32 nRet;
		/*int nYear =*/ GetYearAndRemainder( nRet );
		return nRet;
	}

	inline void	DateTime::AddTime(int nHours, int nMinutes, int nSeconds){
		m_nSeconds += (Int64)nSeconds	+ (60 /*seconds/minute*/ * ((Int64)nMinutes
						+ (60 /*minutes/hour*/ * (Int64)nHours) ) );
	}

	inline void	DateTime::AddDays(int nDays /*= 1*/){ m_nSeconds += (Int64)(time_constants::g_n32SecondsPerDay * nDays); }

	inline void DateTime::Add(int nYears, int nMonths, int nDays, int nHours, int nMinutes, int nSeconds){
		if (nYears) AddYears( nYears );
		if (nMonths) AddMonths( nMonths );
		AddDays( nDays );
		AddTime( nHours, nMinutes, nSeconds );
	}		
	
	inline /*static*/ DateTime	DateTime::GetCurrentDate(){
		int nYear; int nMonth, nDay;
		DateTime   dtNow = Now();
		dtNow.GetDate(nYear, nMonth, nDay);
		return DateTime(nYear, nMonth, nDay, 0, 0, 0, UTC);
	}
	// inline /*static*/ DateTime DateTime::Zero(){ return DateTime(0,1,1, 0,0,0, UTC); }

		/** Operations **/

	inline bool DateTime::operator==( const DateTime& dt ) const { return m_nNanoseconds == dt.m_nNanoseconds && GetUTCSeconds() == dt.GetUTCSeconds(); }
	inline bool DateTime::operator!=( const DateTime& dt ) const { return m_nNanoseconds != dt.m_nNanoseconds || GetUTCSeconds() != dt.GetUTCSeconds(); }
	inline bool DateTime::operator<=( const DateTime& dt ) const { Int64 ThisUTCSeconds = GetUTCSeconds(), ThatUTCSeconds = dt.GetUTCSeconds(); return (ThisUTCSeconds < ThatUTCSeconds || (ThisUTCSeconds == ThatUTCSeconds && m_nNanoseconds <= dt.m_nNanoseconds)); }
	inline bool DateTime::operator<( const DateTime& dt ) const { Int64 ThisUTCSeconds = GetUTCSeconds(), ThatUTCSeconds = dt.GetUTCSeconds(); return (ThisUTCSeconds < ThatUTCSeconds || (ThisUTCSeconds == ThatUTCSeconds && m_nNanoseconds < dt.m_nNanoseconds)); }
	inline bool DateTime::operator>=( const DateTime& dt ) const { Int64 ThisUTCSeconds = GetUTCSeconds(), ThatUTCSeconds = dt.GetUTCSeconds(); return (ThisUTCSeconds > ThatUTCSeconds || (ThisUTCSeconds == ThatUTCSeconds && m_nNanoseconds >= dt.m_nNanoseconds)); }
	inline bool DateTime::operator>( const DateTime& dt ) const { Int64 ThisUTCSeconds = GetUTCSeconds(), ThatUTCSeconds = dt.GetUTCSeconds(); return (ThisUTCSeconds > ThatUTCSeconds || (ThisUTCSeconds == ThatUTCSeconds && m_nNanoseconds > dt.m_nNanoseconds)); }

	inline const DateTime& DateTime::operator=( const DateTime& cp ){ 
		m_nSeconds = cp.m_nSeconds;
		m_nNanoseconds = cp.m_nNanoseconds;
		m_nBias	 = cp.m_nBias;
		return *this;
	}

	inline const DateTime& DateTime::operator+=( const TimeSpan& tmSpan ) { 
		Int32 NewNanoseconds = (Int32)m_nNanoseconds + tmSpan.m_nElapsedNanoseconds;
		if (NewNanoseconds < 0) { m_nSeconds --; NewNanoseconds += time_constants::g_n32NanosecondsPerSecond; } 
		else if (NewNanoseconds >= time_constants::g_n32NanosecondsPerSecond) { m_nSeconds ++; NewNanoseconds -= time_constants::g_n32NanosecondsPerSecond; }
		m_nNanoseconds = NewNanoseconds;
		m_nSeconds += tmSpan.m_nElapsedSeconds;
		return *this;
	}

	inline const DateTime& DateTime::operator-=( const TimeSpan& tmSpan ) {
		Int32 NewNanoseconds = (Int32)m_nNanoseconds - tmSpan.m_nElapsedNanoseconds;
		if (NewNanoseconds < 0) { m_nSeconds --; NewNanoseconds += time_constants::g_n32NanosecondsPerSecond; }
		else if (NewNanoseconds >= time_constants::g_n32NanosecondsPerSecond) { m_nSeconds ++; NewNanoseconds -= time_constants::g_n32NanosecondsPerSecond; }
		m_nNanoseconds = NewNanoseconds;
		m_nSeconds -= tmSpan.m_nElapsedSeconds;
		return *this;
	}

	inline DateTime DateTime::operator+( const TimeSpan& tmSpan ) const {
		DateTime ret(*this);
		ret += tmSpan;		
		return ret;
	}

	inline DateTime DateTime::operator-( const TimeSpan& tmSpan ) const {
		DateTime ret(*this);
		ret -= tmSpan;
		return ret;
	}

	inline TimeSpan DateTime::operator-( const DateTime& b ) const 
	{ 
		TimeSpan ret;		
		// 20 and 400 nanoseconds minus 15 and 300 nanoseconds:		delta = 5 seconds and 100 nanoseconds
		// 20 and 300 nanoseconds minus 15 and 400 nanoseconds:		delta = 4 seconds and 999,999,900 nanoseconds
		// 15 and 300 nanoseconds minus 20 and 400 nanoseconds:		delta = -5 seconds and -100 nanoseconds
		// 15 and 400 nanoseconds minus 20 and 300 nanoseconds:		delta = -4 seconds and -999,999,900 nanoseconds
		ret.m_nElapsedSeconds = GetUTCSeconds() - b.GetUTCSeconds();
		ret.m_nElapsedNanoseconds = m_nNanoseconds - b.m_nNanoseconds;
		if (ret.m_nElapsedSeconds >= 0)
		{
			if (ret.m_nElapsedNanoseconds < 0) { ret.m_nElapsedSeconds --; ret.m_nElapsedNanoseconds += time_constants::g_n32NanosecondsPerSecond; }
			return ret;
		}
		else
		{		
			if (ret.m_nElapsedNanoseconds > 0) { ret.m_nElapsedSeconds ++; ret.m_nElapsedNanoseconds -= time_constants::g_n32NanosecondsPerSecond; }
			return ret;
		}
	}		

		/** Conversions **/	

	inline DateTime::operator time_t() const
	{
		Int64 nValueAsUTC = GetUTCSeconds();
		if (nValueAsUTC < DateTime_g_nOffsetForTimeT) return 0;
		return (time_t)(nValueAsUTC - DateTime_g_nOffsetForTimeT);
	}

	inline void	DateTime::Set(time_t nInput){ 
		m_nSeconds		= ((UInt64)nInput) + DateTime_g_nOffsetForTimeT;
		m_nNanoseconds	= 0;
		m_nBias			= 0;						// time_t values are specified in UTC time.		
	}

	inline void DateTime::Set(UInt64 nSeconds, UInt32 nNanoseconds, int nBiasSeconds){
		m_nSeconds		= nSeconds;
		m_nNanoseconds	= nNanoseconds;
		m_nBias			= nBiasSeconds;
	}

	inline DateTime	DateTime::asUTC() const {
		DateTime	ret;
		ret.m_nSeconds		= GetUTCSeconds();
		ret.m_nNanoseconds	= m_nNanoseconds;
		ret.m_nBias			= 0;
		return ret;
	}

	inline DateTime	DateTime::asLocalTime() const {
		DateTime	ret;
		ret.m_nSeconds		= GetLocalSeconds();
		ret.m_nNanoseconds	= m_nNanoseconds;
		ret.m_nBias			= GetLocalBias();
		return ret;
	}

	inline DateTime::operator tm() const
	{
		tm tmRet;
		memset(&tmRet, 0, sizeof(tmRet));
		UInt32 nRemainder; 
		int nYear = GetYearAndRemainder(nRemainder);
		if (nYear < 1900) throw Exception(S("tm structure cannot represent date range."));
		tmRet.tm_year	= nYear - 1900;
		tmRet.tm_mon    = GetMonthFromRemainder( nRemainder, nYear, IsLeapYear(nYear) ) - 1;
		tmRet.tm_mday	= GetDaysFromRemainder( nRemainder ) + 1;
		tmRet.tm_hour	= GetHoursFromRemainder( nRemainder );
		tmRet.tm_min	= GetMinutesFromRemainder( nRemainder );
		tmRet.tm_sec	= (int)nRemainder;
		DayOfWeek dow;
		GetDayOfWeek( dow );
		tmRet.tm_wday	= (int)dow;
		assert( tmRet.tm_wday >= 0 && tmRet.tm_wday <= 6 );
		return tmRet;
	}	

	inline string	DateTime::Format( const char* lpszFormatStr ) const
	{
		char str[256];
		tm tmValue	= operator tm();
		assert( tmValue.tm_wday >= 0 && tmValue.tm_wday <= 6 );
		strftime(str, 256, lpszFormatStr, &tmValue);
		return str;
	}

	inline string	DateTime::asPresentationString( bool bSeconds /*= true*/ ) const { return bSeconds ? Format( S("%I:%M:%S %p %A %B %d, %Y") ) : Format( S("%I:%M %p %A %B %d, %Y") ); }
	inline string	DateTime::asLongString() const { return Format( S("%I:%M:%S %p on %A, %B %d, %Y") ); }

	inline string	DateTime::asMediumString( bool bSeconds, bool b24Hr, bool bYear ) const { 
		if( bYear )
		{
			if( b24Hr )
			{
				if( bSeconds )	return Format( S("%a %b %d %Y %H:%M:%S") ); 
				else			return Format( S("%a %b %d %Y %H:%M") ); 
			}
			else
			{
				if( bSeconds )	return Format( S("%a %b %d %Y %I:%M:%S %p") ); 
				else			return Format( S("%a %b %d %Y %I:%M %p") ); 
			}
		}
		else
		{
			if( b24Hr )
			{
				if( bSeconds )	return Format( S("%a %b %d %H:%M:%S") ); 
				else			return Format( S("%a %b %d %H:%M") ); 
			}
			else
			{
				if( bSeconds )	return Format( S("%a %b %d %I:%M:%S %p") ); 
				else			return Format( S("%a %b %d %I:%M %p") ); 
			}
		}
	}

	inline string	DateTime::asShortString() const { return Format( S("%b %d %y %H:%M:%S") ); }    
	inline string	DateTime::asNumericString() const { return Format( S("%d.%m.%Y %H:%M:%S") ); }
	inline string	DateTime::asMilString() const { return Format( S("%d%b%y %H:%M:%S") ); }

	inline string	DateTime::asDateString( bool bYear /*= true*/ ) const { 
		if( bYear )
			return Format( S("%A %B %d, %Y") ); 
		else
			return Format( S("%A %B %d") ); 
	}

	inline string	DateTime::asTimeString( bool bSeconds /*= true*/, bool b24Hr /*= true*/ ) const { 
		if( bSeconds ){
			if( b24Hr ) return Format( S("%H:%M:%S") ); 
			else		return Format( S("%I:%M:%S %p") );
		}
		else {
			if( b24Hr ) return Format( S("%H:%M") ); 
			else		return Format( S("%I:%M %p") );
		}
	}

	inline string	DateTime::asDebugString() const { return Format( S("%d.%m.%Y %H.%M.%S") ); }
	inline string	DateTime::asInternetString() const { 
		return asUTC().Format(S("%a, %d %b %Y %H:%M:%S GMT")); 
	}
	inline string	DateTime::ToString() const { return asISO8601(9); }

	inline string	DateTime::asISO8601(int Precision /*= 6*/) const { 

		// Equivalent to:	 string strA = Format(S("%Y-%m-%dT%H:%M:"));
		// but supports dates beyond the 1900-67435 year range.
		int nYear, nMonth, nDay, nHour, nMinute, nSecond;
		Get(nYear, nMonth, nDay, nHour, nMinute, nSecond);
		char pszA[64];
		#if defined(_MSC_VER)
		sprintf_s(pszA, S("%d-%02d-%02dT%02d:%02d:"), nYear, nMonth, nDay, nHour, nMinute);
		#else
		sprintf(pszA, S("%d-%02d-%02dT%02d:%02d:"), nYear, nMonth, nDay, nHour, nMinute);
		#endif		
		string strA = pszA;

		char pszB[64];
		double dSecond = (double)nSecond + (double)m_nNanoseconds * time_constants::g_dSecondsPerNanosecond;
		#if defined(_MSC_VER)
		sprintf_s(pszB, S("%0*.*f"), (Precision + 3), Precision, dSecond);
		#else
		sprintf(pszB, S("%0*.*f"), (Precision + 3), Precision, dSecond);
		#endif

		if (abs(GetBias()) >= 60)
		{
			// Example A: 18:00 in UTC, but current value is 13:40 with bias of -04:20.
			// -04:20 = 4 hours and 20 minutes = - (4 hours * 3600) - (20 minutes * 60) = -15600 seconds.
			// Example B: 18:00 in UTC, but current value is 19:10 with bias of +01:10.
			// +01:10 = 1 hours and 10 minutes = (1 hour * 3600) + (10 minutes * 60) = +4200 seconds.
			int nBias = GetBias();						// Examples: -15600 seconds, +4200 seconds

			int sign = (nBias >= 0) ? 1 : -1;
			int nBiasHours = sign * (abs(nBias) / 3600);		// Examples: -4 hours, 1 hour

			int nRem = abs(nBias) % 3600;		// Remaining seconds after extracting the hours (examples: 1200 which is 20 minutes, 600 which is 10 minutes).
			UInt32 nBiasMinutes = nRem / 60;
			char pszC[64];
			#if defined(_MSC_VER)
			sprintf_s(pszC, S("%+03d:%02u"), nBiasHours, nBiasMinutes);
			#else
			sprintf(pszC, S("%+03d:%02u"), nBiasHours, (uint)nBiasMinutes);
			#endif
			return strA + pszB + pszC;
		}
		else return strA + pszB + S("Z");	
	}

			/** For example, for January returns 31. **/
	inline /*static*/ int DateTime::GetDaysInMonth( int nMonth, bool bLeapYear ){
		assert( nMonth >= 1 && nMonth <= 12 );
		return bLeapYear ? time_constants::g_tableDaysInMonthLY[nMonth] : time_constants::g_tableDaysInMonthNLY[nMonth];
	}
	inline /*static*/ int DateTime::GetDaysInMonth( int nMonth, int nYear ){ return GetDaysInMonth( nMonth, IsLeapYear(nYear) ); }

		/** Core Members **/

	inline /*static*/ DateTime	DateTime::Now()
	{
#	ifdef _WIN32
		SYSTEMTIME	st;
		GetLocalTime(&st);
		DateTime	now(st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds * 1000000, LocalTimeZone);
		return now;
#	elif defined(_LINUX)
		return UtcNow().asLocalTime();
#	else
#		error Platform-specific code is required for retrieving the current date and time as local time.
#	endif
	}

	inline /*static*/ DateTime	DateTime::UtcNow()
	{
#	ifdef _WIN32
		SYSTEMTIME	st;
		GetSystemTime(&st);
		DateTime	now(st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond, st.wMilliseconds * 1000000, UTC);
		return now;
#	elif defined(_LINUX)

		// Note: Program must be linked with -lrt (real-time library) for this support...
		// clock_gettime(), when using CLOCK_REALTIME, measures time relative to the Epoch, which appears to be in UTC and
		// is the same Epoch as time_t uses.
		timespec tp;
		if (clock_gettime(CLOCK_REALTIME, &tp) != 0) Exception::ThrowFromErrno(errno);
		DateTime	now;
		now.m_nSeconds = ((Int64)tp.tv_sec) + g_nOffsetForTimeT;
		now.m_nNanoseconds = tp.tv_nsec;
		now.m_nBias = 0;
		return now;

#	else
#		error Platform-specific code is required for retrieving the current date and time as UTC time.
#	endif
	}

	inline /*static*/ int DateTime::GetLocalBias()
	{
		// See top of this header for some discussion.

		int	nBiasMinutes;
#	ifdef _WIN32
		TIME_ZONE_INFORMATION	tzi;
		ZeroMemory(&tzi, sizeof(tzi));
		switch (GetTimeZoneInformation(&tzi))
		{
		default:
		case TIME_ZONE_ID_UNKNOWN:	nBiasMinutes = tzi.Bias; break;
		case TIME_ZONE_ID_STANDARD:	nBiasMinutes = tzi.Bias + tzi.StandardBias; break;
		case TIME_ZONE_ID_DAYLIGHT:	nBiasMinutes = tzi.Bias + tzi.DaylightBias; break;
		}

		// Definition here is opposite the GetTimeZoneInformation() definition.
		//
		//	GetTimeZoneInformation():		UTC = local time + bias
		//	DateTime:						UTC + bias = local time
		nBiasMinutes = -nBiasMinutes;
#	elif defined(_LINUX)		

		// Get the local time information right now, although all we care about is the GMT offset value.				
		time_t t = time(NULL);
		struct tm lt = { 0 };
		localtime_r(&t, &lt);

		// According to libc docs, tm_gmtoff gives the number of seconds that you must add to UTC to get
		// local time:
		//
		//	UTC + bias = local time
		//	UTC + tm_gmtoff = local time
		//
		// This happens to match my convention here.
		return lt.tm_gmtoff;		// It even happens to be in seconds.  Not sure on portability.

#	else
#		error Platform-specific code is required for retrieving and calculating time zone bias information.
#	endif

		assert(nBiasMinutes > -1440 && nBiasMinutes < 1440);	// Possible range for bias is +/- 23.99 hours.

		return (nBiasMinutes * 60 /* seconds/minute */);

	}// End GetLocalBias()

	inline /*static*/ void DateTime::SetSystemTime(const DateTime& to)
	{
#if defined(_WINDOWS)

		SYSTEMTIME  st = to.asUTC().ToSYSTEMTIME();
		if (!::SetSystemTime(&st)) Exception::ThrowFromWin32(::GetLastError());

#elif defined(_LINUX)

		time_t val = (time_t)to;
		if (stime(&val) == 0) return;
		Exception::ThrowFromErrno(errno);

#endif
	}

#if 0
	void	DateTime::Set(int nYear, int nSeconds, int nBiasMinutes /*= UTC*/)
	{
		m_nSeconds = 0;
		m_nNanoseconds = 0;

		/** If nBiasMinutes is on automatic, determine the local bias **/
		/** Possible Improvement: Base on input date/time instead of on current date/time **/

		if (nBiasMinutes == (int)LocalTimeZone) m_nBias = GetLocalBias();
		else m_nBias = (nBiasMinutes * 60 /* seconds/minute */);

		Add(nYear, /*Month=*/ 0, /*Day=*/ 0, /*Hour=*/ 0, /*Minute=*/ 0, (int)nSeconds);
	}
#endif

	inline void DateTime::Set(int nYear, int nMonth, int nDay, int nHour, int nMinute, int nSecond, int nNanoseconds /*= 0*/, int nBiasMinutes /*= LocalTimeZone*/)
	{
		/** Validate parameters **/

		assert(nMonth >= 1 && nMonth <= 12);
		assert(nDay >= 1 && nDay <= 31);
		assert(nHour >= 0 && nHour <= 23);
		assert(nMinute >= 0 && nMinute <= 59);
		assert(nSecond >= 0 && nSecond <= 59);
		assert(nNanoseconds >= 0 && nNanoseconds < time_constants::g_n32NanosecondsPerSecond);

		/** If nBiasMinutes is on automatic, determine the local bias **/
		/** Possible Improvement: Base on input date/time instead of on current date/time **/

		if (nBiasMinutes == (int)LocalTimeZone) m_nBias = GetLocalBias();
		else m_nBias = (nBiasMinutes * 60 /* seconds/minute */);

		// Assertion: Valid range for bias is +/- 23.99 hours.
		assert(m_nBias > -(24 * 60 * 60) && m_nBias < (24 * 60 * 60));

		/** Time of day and Day of month **/

		// Assertion: There are not that many days in that month, in a leap-year.
		assert(!IsLeapYear(nYear) || nDay <= time_constants::g_tableDaysInMonthLY[nMonth]);
		// Assertion: There are not that many days in that month, in a non-leap-year.
		assert(IsLeapYear(nYear) || nDay <= time_constants::g_tableDaysInMonthNLY[nMonth]);

		m_nSeconds = (Int64)nSecond + (60 /*seconds/minute*/ * ((Int64)nMinute
			+ (60 /*minutes/hour*/ * ((Int64)nHour + (24 /*hours/day*/ * (Int64)(nDay - 1))))));

		/** Months **/

		/* Note: IsLeapYear(void) is not yet available because m_nSeconds is not yet completely calculated. */

		if (IsLeapYear(nYear))
			m_nSeconds += (Int64)time_constants::g_tableDaysPastInYearLY[nMonth] * 60 * 60 * 24;
		else
			m_nSeconds += (Int64)time_constants::g_tableDaysPastInYearNLY[nMonth] * 60 * 60 * 24;

		/** Years **/

		UInt32	nAbsYear = abs(nYear);
		UInt32	nNumberOfLeapYears = 0;
		//		-	The year zero was a leap year.
		//		Also, since we are counting year zero here and the later calculations count years *PAST*, we
		//		subtract one from our working year.
		if (nAbsYear) { nAbsYear--; nNumberOfLeapYears++; }
		//		-	If the year is divisible by 4 but does not end in 00, then the year is a leap year, with 366 days. 
		//		-	If the year is not divisible by 4, then the year is a nonleap year, with 365 days. 
		//			Number of leap years = (Years divided by 4) rounded downward to integer
		//							less   (Years divided by 100) rounded downward to integer
		nNumberOfLeapYears += (nAbsYear / 4u) - (nAbsYear / 100u);
		//		-	If the year ends in 00 but is not divisible by 400, then the year is a nonleap year, with 365 days. 
		//		-	If the year ends in 00 and is divisible by 400, then the year is a leap year, with 366 days. 
		//			Addition leap years  = (Years divided by 400) rounded downward to integer
		nNumberOfLeapYears += (nAbsYear / 400u);

		assert((UInt32)abs(nYear) >= nNumberOfLeapYears);

		// Note: We intentionally revert to using abs(nYear) instead of nAbsYear now.  We need the 0th year back in.
		UInt32	nNumberOfNonLeapYears = abs(nYear) - nNumberOfLeapYears;

#if 0
#	ifdef _DEBUG	/** Debugging: Verify the above calculations by integrating over the 'IsLeapYear()' function... */
		UInt32	nDebugNumberOfLeapYears = 0, nDebugNumberOfNonLeapYears = 0;
		if (nYear >= 0) {
			for (int zi = 0; zi < nYear; zi++)
				if (IsLeapYear(zi)) nDebugNumberOfLeapYears++; else nDebugNumberOfNonLeapYears++;
		}
		else {
			for (int zi = 0; zi > -nYear; zi--)
				if (IsLeapYear(zi)) nDebugNumberOfLeapYears++; else nDebugNumberOfNonLeapYears++;
		}
		assert(nDebugNumberOfLeapYears == nNumberOfLeapYears);
		assert(nDebugNumberOfNonLeapYears == nNumberOfNonLeapYears);
#	endif
#endif

		if (nYear < 0)
		{
			m_nSeconds -= (Int64)nNumberOfLeapYears * 60ll * 60ll * 24ll * 366ll;
			m_nSeconds -= (Int64)nNumberOfNonLeapYears * 60ll * 60ll * 24ll * 365ll;
		}
		else
		{
			m_nSeconds += (Int64)nNumberOfLeapYears * 60ll * 60ll * 24ll * 366ll;
			m_nSeconds += (Int64)nNumberOfNonLeapYears * 60ll * 60ll * 24ll * 365ll;
		}

#ifdef _DEBUG
		assert((3 / 4) == 0);					// Ensure that compiler rounds downward to integer.
		UInt32 nTrial3 = 3, nTrial4 = 4;
		assert((nTrial3 / nTrial4) == 0);		// Ensure that hardware rounds downward to integer.
#endif		

		m_nNanoseconds = nNanoseconds;

	}// End of DateTime::Set()

	inline int DateTime::GetYearAndRemainder(UInt32& nRemainder) const
	{
		/*	Terms I invented to simplify this:
			Leap-Set:			Units of 4-years with 0th year a leap-year.
			Non-Leap Set:		Units of 4-years with no leap-years.
			Leap Century:		Units of 100 years.  i.e. from year 0 to 99, inclusive.
			Non-Leap Century:	Units of 100 years.  i.e. from year 100 to 199, inclusive.
			Periods:			Units of 400 years.  i.e. from year 0 to 399, inclusive.

			Leap Century:
				Includes (25) leap-years (year zero counted, year 100 not counted.)
				(Year 100 is not even present since a century here is 0-99.)
				That's 75 non-leap-years.
				A leap century is 100 years.

			Non-Leap Century:
				Includes (24) leap-years (year zero not counted.)
				That's 76 non-leap-years.
				A non-leap century is 100 years.

			Periods:
				Includes (25+24+24+24) leap-years (year zero counted, years 100, 200, and 300 not counted).
				That's 97 leap-years in a period.
				That's 303 non-leap-years in a period.
				A period is 400 years.
				The second period (years 400-799) behaves identical to the first period, and so on.

			A "leap-century" has one extra leap-year/century from a "non-leap-century".
			A "leap-year" has one extra day from a "non-leap-year".

			Outline:
				Any one Period.
					Leap Century.
						25 Leap-Sets { Each 1 Leap-Year 3 Non-Leap-Years }
					3x Non-Leap Centuries.
						Each {
							1 Non-Leap Set (4 Non-Leap-Years)
							24 Leap-Sets { Each 1 Leap-Year 3 Non-Leap-Years }
						}

			Footnote:			An unsigned 32-bit integer can hold the number of seconds in 135+ years.
		*/
		static const Int64 nSecondsPerPeriod
#                               if 0
			= 60i64 /*seconds/minute*/ * 60i64 /*minutes/hour*/ * 24i64 /*hours/day*/
			* ((365i64 /*days*/ * 303i64 /*non-leap-years/period*/)
				+ (366i64 /*days*/ * 97i64 /*leap-years/period*/)) /*days/period*/;
#                               endif
		= 12622780800ll;

		static const UInt64 nSecondsPerLeapCentury
#                               if 0
			= 60i64 /*seconds/minute*/ * 60i64 /*minutes/hour*/ * 24i64 /*hours/day*/
			* ((365i64 /*days*/ * 75i64 /*non-leap-years/leap-century*/)
				+ (366i64 /*days*/ * 25i64 /*leap-years/leap-century*/)) /*days/period*/;
#                               endif
		= 3155760000ull;

		static const UInt64	nSecondsPerNonLeapCentury
#                               if 0
			= 60i64 /*seconds/minute*/ * 60i64 /*minutes/hour*/ * 24i64 /*hours/day*/
			* ((365i64 /*days*/ * 76i64 /*non-leap-years/non-leap-century*/)
				+ (366i64 /*days*/ * 24i64 /*leap-years/non-leap-century*/)) /*days/period*/;
#                               endif                                
		= 3155673600ull;

		static const UInt32	nSecondsPerLeapSet
			= 60 /*seconds/minute*/ * 60 /*minutes/hour*/ * 24 /*hours/day*/
			* ((365 /*days*/ * 3 /*non-leap-years/leap-set*/)
				+ (366 /*days*/ /*x 1 leap-years/leap-set*/)) /*days/period*/;

		static const UInt32	nSecondsPerNonLeapSet
			= 60 /*seconds/minute*/ * 60 /*minutes/hour*/ * 24 /*hours/day*/
			* (365 /*days*/ * 4 /*non-leap-years/non-leap-set*/)/*days/period*/;

		Int64 nYear = 0;
		UInt64 nRemain;
		if (m_nSeconds < 0) nRemain = (UInt64)(-m_nSeconds);
		else nRemain = (UInt64)m_nSeconds;

		/*
			Algorithm:
				Take off number of periods (and count 400 years for each).
				This leaves us *starting* at a year which is evenly divisible by 400.

				We now have as much as this:
					Leap-Century (year 400.)
					3 x Non-Leap Centuries (years 500-799.)

			Data Types:  The maximum m_nSeconds value, Int64_MaxValue / nSecondsPerPeriod is 7.3069e+08,
				which is less than the maximum 32-bit signed integer value of 2.1475e+09.  Therefore, the
				maximum 64-bit value is still supported by capturing periods in a 32-bit value, however
				the maximum resulting year (after x400) is 2.9228e+11, which requires a 64-bit value.
		*/

		UInt32 nNumberOfWholePeriods = (UInt32)(nRemain / nSecondsPerPeriod);		/* Gives units of 'periods' */
		nYear += nNumberOfWholePeriods * 400ll;
		nRemain -= (Int64)nNumberOfWholePeriods * nSecondsPerPeriod;

		/*
			Algorithm B:
			------------
				Any whole leap-centuries?
				Yes.	Count off 1 whole leap-century if present.
						Count off remaining whole non-leap-centuries (up to 2 possible.)
						Any whole non-leap sets?
						Yes.	Count off 1 whole non-leap-set if present.
								Continue at algorithm C below.
						No.		Count off remaining whole non-leap-years.
								Done.

				No.		Continue at algorithm C below.
		*/

		if (nRemain >= nSecondsPerLeapCentury)
		{
			/* "Yes" codition in above algorithm */

			/* Count off whole leap-century */
			nYear += 100;
			nRemain -= nSecondsPerLeapCentury;

			/* Count off whole non-leap-centuries (up to 2 whole are possible, 1 partial.) */
			assert(nRemain < (2 * nSecondsPerNonLeapCentury) + nSecondsPerNonLeapCentury);
			while (nRemain >= nSecondsPerNonLeapCentury)
			{
				nYear += 100;
				nRemain -= nSecondsPerNonLeapCentury;
			}

			/* Left in remain:  0 to 99 years, in a non-leap-century */
			assert(nRemain <= (UInt64)UInt32_MaxValue);
			nRemainder = (UInt32)nRemain;				// Switch to 32-bit arithmetic.

				/* Count off 1 whole non-leap-set if present */
			if (nRemainder >= nSecondsPerNonLeapSet) {
				nYear += 4;
				nRemainder -= nSecondsPerNonLeapSet;
			}
			else
			{
				/* Count off remaining non-leap-years if present.
					Since we are in a partial non-leap-set, 3 whole are possible, and 1 partial.
				*/
				assert(nRemainder < (3 * time_constants::g_nSecondsPerNonLeapYear) + time_constants::g_nSecondsPerNonLeapYear);
				UInt32 nNumberOfNonLeapYears = nRemainder / time_constants::g_nSecondsPerNonLeapYear;
				nYear += nNumberOfNonLeapYears;
				nRemainder -= nNumberOfNonLeapYears * time_constants::g_nSecondsPerNonLeapYear;

				// The remainder should contain no more than 1 year's worth of seconds.
				assert(nRemainder <= time_constants::g_nSecondsPerLeapYear);

				// The internal representation can support years beyond the 32-bit range, but the public interface does not.
				assert(nYear >= Int32_MinValue && nYear <= Int32_MaxValue);

				if (m_nSeconds >= 0) return (int)nYear;
				else return (int)-nYear;
			}

			/* Next:	Count off remaining leap-sets if present.
						Since we are in a non-leap-century, and we've counted off one, 23 are possible.
						1 partial is possible. */
			assert(nRemainder < (23 * nSecondsPerLeapSet) + nSecondsPerLeapSet);
		}
		else {
			/* Else, next:	Count off remaining leap-sets if present.
							Since we are in a leap-century, 25 are possible.  1 partial is possible. */
			assert(nRemain < (25 * nSecondsPerLeapSet) + nSecondsPerLeapSet);

			/* Left in remain:  0 to 99 years, in a leap-century */
			assert(nRemain <= (UInt64)UInt32_MaxValue);
			nRemainder = (UInt32)nRemain;				// Switch to 32-bit arithmetic.
		}

		/**
			Algorithm C
			-----------
				Count off remaining whole leap-sets if present.
				Continue at Algorithm D below.
		**/

		/* Count off remaining leap-sets if present.
		*/
		UInt32 nNumberOfLeapSets = nRemainder / nSecondsPerLeapSet;
		nYear += nNumberOfLeapSets * 4;
		nRemainder -= nNumberOfLeapSets * nSecondsPerLeapSet;

		/**
			Algorithm D
			-----------
				Is there a whole leap-year?
				Yes.	Count off 1 leap-year.
						Count off remaining whole non-leap-years.
						Done.
				No.		Done.
		**/

		/* Count off whole leap-year if present. */
		if (nRemainder >= time_constants::g_nSecondsPerLeapYear)
		{
			nYear++;
			nRemainder -= time_constants::g_nSecondsPerLeapYear;

			/* Count off remaining non-leap-years if present.
				Since we are in a leap-set, and we've counted off one, 2 are possible.
				1 partial is possible.
			*/
			assert(nRemainder < (2 * time_constants::g_nSecondsPerNonLeapYear) + time_constants::g_nSecondsPerNonLeapYear);
			UInt32 nNumberOfNonLeapYears = nRemainder / time_constants::g_nSecondsPerNonLeapYear;
			nYear += nNumberOfNonLeapYears;
			nRemainder -= nNumberOfNonLeapYears * time_constants::g_nSecondsPerNonLeapYear;
		}

		// The remainder should contain no more than 1 year's worth of seconds.
		assert(nRemainder <= time_constants::g_nSecondsPerLeapYear);

		// The internal representation can support years beyond the 32-bit range, but the public interface does not.
		assert(nYear >= Int32_MinValue && nYear <= Int32_MaxValue);

		if (m_nSeconds >= 0) return (int)nYear;
		else return (int)-nYear;

	}// End of GetYearAndRemainder()

	inline int DateTime::GetMonthFromRemainder(UInt32& nRemainder, int nYear, bool bLeapYear) const
	{
		assert((bLeapYear || nRemainder < time_constants::g_nSecondsPerNonLeapYear));
		assert((!bLeapYear || nRemainder < time_constants::g_nSecondsPerLeapYear));

		if (nYear < 0) {
			// At this point, flip the remainder within 1 year.  For example,
			// January 10th, 1 B.C. will have a remainder that corresponds to
			// December 21st, 1 B.C. if the years are negative.  So if we
			// subtract from seconds-in-1-year, we correct for this.
			if (bLeapYear) nRemainder = time_constants::g_nSecondsPerLeapYear - nRemainder;
			else nRemainder = time_constants::g_nSecondsPerNonLeapYear - nRemainder;
		}

		/** Perform a binary search **/

		int nMonth = 6, nAbsDelta = 3;
		for (;;)
		{
			if (nMonth <= 1) return nMonth;

			if (bLeapYear)
			{
				if (nMonth >= 12) {
					nRemainder -= time_constants::g_tableSecondsPastInLeapYear[nMonth];
					return nMonth;
				}

				if (nRemainder >= time_constants::g_tableSecondsPastInLeapYear[nMonth]) {
					if (nRemainder < time_constants::g_tableSecondsPastInLeapYear[nMonth + 1]) {
						nRemainder -= time_constants::g_tableSecondsPastInLeapYear[nMonth];
						return nMonth;
					}
					else nMonth += nAbsDelta;
				}
				else nMonth -= nAbsDelta;
			}
			else
			{
				if (nMonth >= 12) {
					nRemainder -= time_constants::g_tableSecondsPastInNonLeapYear[nMonth];
					return nMonth;
				}

				if (nRemainder >= time_constants::g_tableSecondsPastInNonLeapYear[nMonth]) {
					if (nRemainder < time_constants::g_tableSecondsPastInNonLeapYear[nMonth + 1]) {
						nRemainder -= time_constants::g_tableSecondsPastInNonLeapYear[nMonth];
						return nMonth;
					}
					else nMonth += nAbsDelta;
				}
				else nMonth -= nAbsDelta;
			}

			nAbsDelta >>= 1;
			if (!nAbsDelta) nAbsDelta = 1;
		}
	}// End of GetMonthFromRemainder()

	inline void	DateTime::AddMonths(int nMonths /*= 1*/)
	{
		UInt32 nRemainder;
		int nYear = GetYearAndRemainder(nRemainder);
		bool bLeapYear = IsLeapYear(nYear);
		int nThisMonth = GetMonthFromRemainder(nRemainder, nYear, bLeapYear);

		if (nMonths > 0)
		{
			while (nMonths--)
			{
				if (nThisMonth == 12)
				{
					if (bLeapYear)
						m_nSeconds += time_constants::g_tableSecondsInMonthLY[nThisMonth];
					else
						m_nSeconds += time_constants::g_tableSecondsInMonthNLY[nThisMonth];

					nYear++;
					nThisMonth = 1;
					bLeapYear = IsLeapYear(nYear);
					continue;
				}

				if (bLeapYear)
					m_nSeconds += time_constants::g_tableSecondsInMonthLY[nThisMonth];
				else
					m_nSeconds += time_constants::g_tableSecondsInMonthNLY[nThisMonth];
				nThisMonth++;
			}
		}
		else
		{
			nMonths = abs(nMonths);
			while (nMonths--)
			{
				if (nThisMonth == 1)
				{
					if (bLeapYear)
						m_nSeconds -= time_constants::g_tableSecondsInMonthLY[nThisMonth];
					else
						m_nSeconds -= time_constants::g_tableSecondsInMonthNLY[nThisMonth];

					nYear--;
					nThisMonth = 12;
					bLeapYear = IsLeapYear(nYear);
					continue;
				}

				if (bLeapYear)
					m_nSeconds -= time_constants::g_tableSecondsInMonthLY[nThisMonth];
				else
					m_nSeconds -= time_constants::g_tableSecondsInMonthNLY[nThisMonth];
				nThisMonth--;
			}
		}
	}// End of AddMonths()

	inline void DateTime::AddYears(int nYears /*= 1*/)
	{
		if (nYears >= 0)
		{
			while (nYears--)
			{
				if (IsLeapYear())
					m_nSeconds += time_constants::g_nSecondsPerLeapYear;
				else
					m_nSeconds += time_constants::g_nSecondsPerNonLeapYear;
			}
		}
		else
		{
			nYears = abs(nYears);
			while (nYears--)
			{
				if (IsLeapYear())
					m_nSeconds -= time_constants::g_nSecondsPerLeapYear;
				else
					m_nSeconds -= time_constants::g_nSecondsPerNonLeapYear;
			}
		}
	}// End of AddYears()

			/** String Functions **/

	inline string DTParse_getWord(const string& str)
	{
		string word;
		for (size_t ii = 0; ii < str.length(); ii++) if (str[ii] == ' ') return word; else word += str[ii];
		return word;
	}

	inline /*static*/ bool DateTime::TryParse(const char* psz, DateTime& Value)
	{
		string str = psz;

		/**
				Sun, 06 Nov 1994 08:49:37 GMT		; (1) RFC 822, updated by RFC 1123
				Sunday, 06-Nov-94 08:49:37 GMT		; (2) RFC 850, obsoleted by RFC 1036
				Sun Nov  6 08:49:37 1994			; (3) ANSI C's asctime() format
				1:13:15 a.m. Sunday June 8, 2005	; (4) asPresentationString() format
				1994-11-05T08:15:30-05:00			; (5a) ISO 8601 with a bias
				1994-11-05T08:15:30Z				; (5b) ISO 8601 with UTC

				012345678901234567890123456789
		**/

		// Only formats 1 (returned by ToString()), 2, 4, and 5 are currently supported.

		if (str.find(S("a.m.")) != string::npos || str.find(S("p.m.")) != string::npos || str.find(S("A.M.")) != string::npos || str.find(S("P.M.")) != string::npos
			|| str.find(S("pm")) != string::npos || str.find(S("am")) != string::npos || str.find(S("PM")) != string::npos || str.find(S("AM")) != string::npos)
		{
			// Format 4:   01:13:15 a.m. Sunday June 08, 2005

			size_t iIndex;

			// Read hours...
			Int32 nHour;
			if (!Int32_TryParse(str.c_str(), NumberStyles::Integer, nHour)) return false;
			if (str[1] == ':') iIndex = 2;
			else if (str[2] == ':') iIndex = 3;
			else return false;

			// Read minutes...
			if (iIndex + 2 >= str.length()) return false;
			Int32 nMinute;
			if (!Int32_TryParse(str.substr(iIndex, 2).c_str(), NumberStyles::Integer, nMinute)) return false;
			iIndex += 2;

			// Read seconds, if present
			Int32 nSecond = 0;
			if (iIndex >= str.length()) return false;
			if (str[iIndex] == ':') {
				iIndex++;
				if (iIndex + 2 >= str.length()) return false;
				if (!Int32_TryParse(str.substr(iIndex, 2).c_str(), NumberStyles::Integer, nSecond)) return false;
				iIndex += 2;
			}
			while (iIndex < str.length() && str[iIndex] == ' ') iIndex++;		// Skip whitespace

				// Read seconds & am/pm...
			bool bMorning;
			if (iIndex + 5 >= str.length()) return false;
			if (to_lower(str.substr(iIndex, 4)).compare(S("a.m.")) == 0) { bMorning = true; iIndex += 5; }
			else if (to_lower(str.substr(iIndex, 4)).compare(S("p.m.")) == 0) { bMorning = false; iIndex += 5; }
			else if (to_lower(str.substr(iIndex, 2)).compare(S("am")) == 0) { bMorning = true; iIndex += 3; }
			else if (to_lower(str.substr(iIndex, 2)).compare(S("pm")) == 0) { bMorning = false; iIndex += 3; }
			else return false;

			// Skip over weekday...
			if (iIndex >= str.length()) return false;
			string	strWeekday = DTParse_getWord(str.substr(iIndex));
			iIndex += strWeekday.length() + 1;

			// Read month...
			int nMonth;
			if (iIndex >= str.length()) return false;
			string strMonth = to_lower(DTParse_getWord(str.substr(iIndex)));
			if (strMonth.compare(S("january")) == 0)			nMonth = 1;
			else if (strMonth.compare(S("february")) == 0)	nMonth = 2;
			else if (strMonth.compare(S("march")) == 0)		nMonth = 3;
			else if (strMonth.compare(S("april")) == 0)		nMonth = 4;
			else if (strMonth.compare(S("may")) == 0)			nMonth = 5;
			else if (strMonth.compare(S("june")) == 0)		nMonth = 6;
			else if (strMonth.compare(S("july")) == 0)		nMonth = 7;
			else if (strMonth.compare(S("august")) == 0)		nMonth = 8;
			else if (strMonth.compare(S("september")) == 0)	nMonth = 9;
			else if (strMonth.compare(S("october")) == 0)		nMonth = 10;
			else if (strMonth.compare(S("november")) == 0)	nMonth = 11;
			else if (strMonth.compare(S("december")) == 0)	nMonth = 12;
			else return false;
			iIndex += strMonth.length() + 1;

			// Read day of month and comma
			if (iIndex + 2 >= str.length()) return false;
			Int32 nDay;
			if (!Int32_TryParse(str.substr(iIndex, 2).c_str(), NumberStyles::Integer, nDay)) return false;
			while (iIndex < str.length() && str[iIndex] != ',') iIndex++;
			if (iIndex + 1 >= str.length() || str[iIndex] != ',') return false;
			iIndex++;
			if (iIndex + 1 >= str.length() || str[iIndex] != ' ') return false;
			iIndex++;

			// Read year
			Int32 nYear;
			if (!Int32_TryParse(str.substr(iIndex).c_str(), NumberStyles::Integer, nYear)) return false;

			// Apply modifiers
			if (!bMorning) {
				if (nHour != 12) nHour += 12;		// 12 p.m. becomes '12' on tweny-four hour clock.  All other p.m. hours get +12.
			}
			else {
				if (nHour == 12) nHour = 0;		// 12 a.m. becomes '0' on twenty-four hour clock.  All other a.m. hours are unmodified.
			}

			if ((nMonth >= 1 && nMonth <= 12)
				&& (nDay >= 1 && nDay <= 31)
				&& (nHour >= 0 && nHour <= 23)
				&& (nMinute >= 0 && nMinute <= 59)
				&& (nSecond >= 0 && nSecond <= 59))
			{
				// Presentation time is always assumed to be given in the local time zone.

				Value = DateTime(nYear, nMonth, nDay, nHour, nMinute, nSecond, 0, LocalTimeZone);
				return true;
			}
			else return false;
		}
		else if (str.find(S("GMT")) != string::npos || str.find(S("UTC")) != string::npos)
		{
			// Sun, 06 Nov 1994 08:49:37 GMT		; (1) RFC 822, updated by RFC 1123
			// Sunday, 06-Nov-94 08:49:37 GMT		; (2) RFC 850, obsoleted by RFC 1036
			// Sun, 06 Nov 1994 08:49:37 UTC		; Supported variant of (1)
			// Sunday, 06-Nov-94 08:49:37 UTC		; Supported variant of (2)

			size_t iIndex = str.find(',');
			if (iIndex == string::npos) return false;

			while (iIndex < str.length() && (str[iIndex] == ',' || isspace(str[iIndex]))) iIndex++;
			if (iIndex >= str.length()) return false;
			if (!isdigit(str[iIndex])) return false;

			Int32 nDay;
			if (!Int32_TryParse(str.substr(iIndex).c_str(), NumberStyles::Integer, nDay)) return false;
			while (iIndex < str.length() && !isalpha(str[iIndex])) iIndex++;
			if (iIndex >= str.length()) return false;

			// Note that extended month names (i.e. January) will also work since the first 3 letters are
			// always the abreviation.
			int nMonth;
			string strMonth = to_lower(str.substr(iIndex, 3));
			if (strMonth.compare(S("jan")) == 0) nMonth = 1;
			else if (strMonth.compare(S("feb")) == 0) nMonth = 2;
			else if (strMonth.compare(S("mar")) == 0) nMonth = 3;
			else if (strMonth.compare(S("apr")) == 0) nMonth = 4;
			else if (strMonth.compare(S("may")) == 0) nMonth = 5;
			else if (strMonth.compare(S("jun")) == 0) nMonth = 6;
			else if (strMonth.compare(S("jul")) == 0) nMonth = 7;
			else if (strMonth.compare(S("aug")) == 0) nMonth = 8;
			else if (strMonth.compare(S("sep")) == 0) nMonth = 9;
			else if (strMonth.compare(S("oct")) == 0) nMonth = 10;
			else if (strMonth.compare(S("nov")) == 0) nMonth = 11;
			else if (strMonth.compare(S("dec")) == 0) nMonth = 12;
			else return false;

			// Skip past spaces, dashes, and month name until we get to the year.
			while (iIndex < str.length() && !isnumeric(str[iIndex])) iIndex++;
			if (iIndex >= str.length()) return false;
			Int32 nYear;
			if (!Int32_TryParse(str.substr(iIndex).c_str(), NumberStyles::Integer, nYear)) return false;

			if (nYear < 100) {
				DateTime dtNow = DateTime::Now();
				Int32 nCurrentYear = dtNow.GetYear();
				Int32 nCurrent2Year = nCurrentYear % 100ll;
				Int32 nCurrentCentury = (nCurrentYear - nCurrent2Year);
				nYear = nCurrentCentury + nYear;
			}

			// Skip to the whitespace following the year.
			while (iIndex < str.length() && !isspace(str[iIndex])) iIndex++;
			if (iIndex >= str.length()) return false;
			// Skip to the first numeric following the whitespace following the year.
			while (iIndex < str.length() && !isnumeric(str[iIndex])) iIndex++;
			if (iIndex >= str.length()) return false;

			str = str.substr(iIndex);

			// 08:49:37 GMT
			// 012345678901

			string strGMT = str.substr(9, 3);
			if (str.length() < 12 || (strGMT.compare(S("GMT")) != 0 && strGMT.compare(S("UTC")) != 0)
				|| str[2] != ':' || str[5] != ':') return false;

			string strHour = str.substr(0, 2);
			string strMin = str.substr(3, 2);
			string strSec = str.substr(6, 2);

			Int32 nHour, nMinute, nSecond;
			if (!Int32_TryParse(strHour, NumberStyles::Integer, nHour)
				|| !Int32_TryParse(strMin, NumberStyles::Integer, nMinute)
				|| !Int32_TryParse(strSec, NumberStyles::Integer, nSecond)) return false;

			if ((nMonth >= 1 && nMonth <= 12)
				&& (nDay >= 1 && nDay <= 31)
				&& (nHour >= 0 && nHour <= 23)
				&& (nMinute >= 0 && nMinute <= 59)
				&& (nSecond >= 0 && nSecond <= 59))
			{
				// This format always stores time in GMT (a.k.a. UTC or Zulu) time.

				Value = DateTime(nYear, nMonth, nDay, nHour, nMinute, nSecond, 0, UTC);
				return true;
			}
			else return false;
		}
		else if (str.length() >= 10 && str[4] == '-' && str[7] == '-')
		{
			// 012345678901234567890123456
			// 1994-11-05T08:15:30-05:00			; (5a) ISO 8601 with a bias
			// 1994-11-05T08:15:30Z					; (5b) ISO 8601 with UTC
			// 1994-11-05T08:15:30.123456-05:00		; (5c) ISO 8601 variant with a bias
			// 1994-11-05T08:15:30.123456Z			; (5d) ISO 8601 variant with UTC

			string strYear = str.substr(0, 4);
			string strMonth = str.substr(5, 2);
			string strDay = str.substr(8, 2);

			Int32 nYear, nMonth, nDay;
			if (!Int32_TryParse(strYear, NumberStyles::Integer, nYear)
				|| !Int32_TryParse(strMonth, NumberStyles::Integer, nMonth)
				|| !Int32_TryParse(strDay, NumberStyles::Integer, nDay)) return false;
			if (nMonth < 1 || nMonth > 12 || nDay < 1 || nDay > 31) return false;

			Int32 nHour = 0, nMinute = 0;
			double dSecond = 0.0;
			Int32 nBiasMinutes = 0;

			if (str.length() >= 16 && str[10] == 'T' && str[13] == ':')
			{
				string strHour = str.substr(11, 2);
				string strMinute = str.substr(14, 2);

				if (!Int32_TryParse(strHour, NumberStyles::Integer, nHour)
					|| !Int32_TryParse(strMinute, NumberStyles::Integer, nMinute)) return false;
				if (nHour < 0 || nHour > 23 || nMinute < 0 || nMinute > 59) return false;

				if (str.length() >= 18)
				{
					size_t nSecondDigits = 0;
					while (17 + nSecondDigits < str.length() && (isdigit(str[17 + nSecondDigits]) || str[17 + nSecondDigits] == '.')) nSecondDigits++;

					if (!nSecondDigits) return false;
					string strSecond = str.substr(17, nSecondDigits);
					if (!Double_TryParse(strSecond, NumberStyles::Float, dSecond)) return false;
					if (dSecond < 0.0 || dSecond > 60.0) return false;

					if (str.length() > 17 + nSecondDigits) {
						if (str[17 + nSecondDigits] == 'Z') nBiasMinutes = 0;
						else if (str[17 + nSecondDigits] == '-' || str[17 + nSecondDigits] == '+')
						{
							if (str.length() >= 17 + nSecondDigits + 3)
							{
								string strBiasHours = str.substr(17 + nSecondDigits, 3);

								Int32 nBiasHours;
								if (!Int32_TryParse(strBiasHours, NumberStyles::Integer, nBiasHours)) return false;

								if (str.length() > 17 + nSecondDigits + 3 && str[17 + nSecondDigits + 3] == ':')
								{
									string strBiasMinutes = str.substr(17 + nSecondDigits + 4, 2);

									if (!Int32_TryParse(strBiasMinutes, NumberStyles::Integer, nBiasMinutes)) return false;

									if (nBiasHours >= 0) nBiasMinutes = nBiasHours * 60 + nBiasMinutes;
									else nBiasMinutes = nBiasHours * 60 - nBiasMinutes;
								}
							}
							else return false;
						}
						else return false;
					}
				}
			}

			Int32 nSecond = (Int32)dSecond;
			Int32 nNanosecond = (Int32)(fmod(dSecond, 1.0) / time_constants::g_dSecondsPerNanosecond);

			Value = DateTime(nYear, nMonth, nDay, nHour, nMinute, nSecond, nNanosecond, nBiasMinutes);
			return true;
		}
		else if (str.length() == 24 && str[3] == ' ' && str[7] == ' ' && str[10] == ' ' && str[19] == ' '
			&& str[13] == ':' && str[16] == ':')
		{
			// Sun Nov  6 08:49:37 1994			; (3) ANSI C's asctime() format
			// 012345678901234567890123
			//			 1		   2

			int nMonth;
			string strMonth = to_lower(str.substr(4, 3));
			if (strMonth.compare(S("jan")) == 0) nMonth = 1;
			else if (strMonth.compare(S("feb")) == 0) nMonth = 2;
			else if (strMonth.compare(S("mar")) == 0) nMonth = 3;
			else if (strMonth.compare(S("apr")) == 0) nMonth = 4;
			else if (strMonth.compare(S("may")) == 0) nMonth = 5;
			else if (strMonth.compare(S("jun")) == 0) nMonth = 6;
			else if (strMonth.compare(S("jul")) == 0) nMonth = 7;
			else if (strMonth.compare(S("aug")) == 0) nMonth = 8;
			else if (strMonth.compare(S("sep")) == 0) nMonth = 9;
			else if (strMonth.compare(S("oct")) == 0) nMonth = 10;
			else if (strMonth.compare(S("nov")) == 0) nMonth = 11;
			else if (strMonth.compare(S("dec")) == 0) nMonth = 12;
			else return false;

			string strDay = str.substr(8, 2);
			string strHour = str.substr(11, 2);
			string strMin = str.substr(14, 2);
			string strSec = str.substr(17, 2);
			string strYear = str.substr(20);

			Int32 nDay, nYear, nHour, nMinute, nSecond;
			if (!Int32_TryParse(strDay, NumberStyles::Integer, nDay)
				|| !Int32_TryParse(strYear, NumberStyles::Integer, nYear)
				|| !Int32_TryParse(strHour, NumberStyles::Integer, nHour)
				|| !Int32_TryParse(strMin, NumberStyles::Integer, nMinute)
				|| !Int32_TryParse(strSec, NumberStyles::Integer, nSecond)) return false;

			if ((nMonth >= 1 && nMonth <= 12)
				&& (nDay >= 1 && nDay <= 31)
				&& (nHour >= 0 && nHour <= 23)
				&& (nMinute >= 0 && nMinute <= 59)
				&& (nSecond >= 0 && nSecond <= 59))
			{
				// This format always stores time in local time.

				Value = DateTime(nYear, nMonth, nDay, nHour, nMinute, nSecond, 0, LocalTimeZone);
				return true;
			}
			else return false;
		}
		else
		{
			return false;					// Unrecognized time format.
		}

	}// End of DateTime::TryParse()

	inline /*static*/ DateTime DateTime::Parse(const char* psz)
	{
		DateTime ret;
		if (!TryParse(psz, ret))
			throw FormatException(S("Unable to parse date/time."));
		return ret;
	}

	inline /*static*/ DateTime DateTime::FromMSDOS(UInt16 date, UInt16 time)
	{
		/**
		* Bitfields for file time:
		*	Bit(s)	Description
		*	15-11	hours (0-23)
		*	10-5	minutes
		*	4-0	seconds/2
		*/

		/**
		* Bitfields for file date:
		*	Bit(s)	Description
		*	15-9	year - 1980
		*	8-5		month
		*	4-0		day
		*/

		unsigned int second = (time & 0x1F) * 2;
		unsigned int minute = (time >> 5) & 0x3F;
		unsigned int hour = (time >> 11) & 0x1F;

		unsigned int day = (date & 0x1F);
		unsigned int month = (date >> 5) & 0x0F;
		unsigned int year = ((date >> 9) & 0x7F) + 1980;

		return DateTime(year, month, day, hour, minute, second);		
	}

	inline void DateTime::asMSDOS(UInt16& date, UInt16& time) const
	{
		date = (((GetYear() - 1980) & 0x7F) << 9);
		date |= ((GetMonth() & 0x0F) << 5);
		date |= (GetDay() & 0x1F);

		time = ((GetHour() & 0x1F) << 11);
		time |= ((GetMinute() & 0x3F) << 5);
		time |= ((GetSecond() >> 1) & 0x1F);
	}
}

#endif	// __DateTime_h__

//	End of DateTime.h


