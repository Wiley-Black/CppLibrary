/*	TimeConstants.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __TimeConstants_h__
#define __TimeConstants_h__

#include "../Platforms/Platforms.h"

namespace wb
{
	namespace time_constants
	{
			/**** Constants needed for elapsed time ****/

		static const Int64	g_nSecondsPerMinute			= 60ll /*seconds/minute*/;
		static const Int64	g_nSecondsPerHour			= 60ll /*minutes/hour*/ * g_nSecondsPerMinute /*seconds/minute*/;
		static const Int64	g_nSecondsPerDay			= 24ll /*hours/day*/ * g_nSecondsPerHour /*seconds/hour*/;
		static const Int64	g_nApproxSecondsPerMonth	= 31ll /*days/month*/ * g_nSecondsPerDay /*seconds/day*/;
		static const Int64	g_nApproxSecondsPerYear		= 12ll /*months/year*/ * g_nApproxSecondsPerMonth /*seconds/month*/;
		static const Int64	g_nMonthsPerYear			= 12ll;
		static const Int64  g_nNanosecondsPerSecond		= 1000000000ll;

			// 32-bit constants...
		static const Int32	g_n32SecondsPerMinute		= 60 /*seconds/minute*/;
		static const Int32	g_n32SecondsPerHour			= 60 /*minutes/hour*/ * g_n32SecondsPerMinute /*seconds/minute*/;
		static const Int32	g_n32SecondsPerDay			= 24 /*hours/day*/ * g_n32SecondsPerHour /*seconds/hour*/;
		static const Int32	g_n32ApproxSecondsPerMonth	= 31 /*days/month*/ * g_n32SecondsPerDay /*seconds/day*/;
		static const Int32	g_n32ApproxDaysPerMonth		= 31;
		static const Int32	g_n32NanosecondsPerSecond	= 1000000000;

			// double-precision constants...
		static const double	g_dSecondsPerMinute			= 60.0 /*seconds/minute*/;
		static const double	g_dSecondsPerHour			= 60.0 /*minutes/hour*/ * g_dSecondsPerMinute /*seconds/minute*/;
		static const double	g_dSecondsPerDay			= 24.0 /*hours/day*/ * g_dSecondsPerHour /*seconds/hour*/;
		static const double	g_dApproxSecondsPerMonth	= 31.0 /*days/month*/ * g_dSecondsPerDay /*seconds/day*/;
		static const double	g_dApproxSecondsPerYear		= 12.0 /*months/year*/ * g_dApproxSecondsPerMonth /*seconds/month*/;			
		static const double g_dSecondsPerNanosecond		= 1.0e-9;

			/**** Constants needed for absolute time ****/					

			// Note: In the following tables, 1=January.  0 is not used.

			// # of days past in year, given start of a month (non-leap-year)			
		static const int g_tableDaysPastInYearNLY[13] = {
			-1, /*January 1st=*/ 0, 31, 59, 90, /*May 1st=*/ 120, 151, 181, 212, /*September 1st=*/ 243, 273, 304, 334
		};

			// # of days past in year, given start of a month (leap-year)
		static const int g_tableDaysPastInYearLY[13] = {
			-1, /*January 1st=*/ 0, 31, 60, 91, /*May 1st=*/ 121, 152, 182, 213, /*September 1st=*/ 244, 274, 305, 335
		};

			// # of days in given months (non-leap-year)
		static const int g_tableDaysInMonthNLY[13] = { 
			-1, /*January*/ 31, 28, 31, 30, /*May=*/ 31, 30, 31, 31, /*September=*/ 30, 31, 30, 31 
		};

			// # of days in given months (leap-year)
		static const int g_tableDaysInMonthLY[13] = { 
			-1, /*January*/ 31, 29, 31, 30, /*May=*/ 31, 30, 31, 31, /*September=*/ 30, 31, 30, 31 
		};

		static const UInt32 g_nSecondsPerLeapYear		= 60 /*seconds/minute*/ * 60 /*minutes/hour*/ * 24 /*hours/day*/
																* ( 366 /*days/leap-year*/ )/*days/period*/;

		static const UInt32 g_nSecondsPerNonLeapYear	= 60 /*seconds/minute*/ * 60 /*minutes/hour*/ * 24 /*hours/day*/
																* ( 365 /*days/non-leap-year*/ )/*days/period*/;

		#define PERDAY	(60 /*seconds/minute*/ * 60 /*minutes/hour*/ * 24 /*hours/day*/)

		// # of seconds past in year, given start of a month
		static const UInt32 g_tableSecondsPastInNonLeapYear[13] = { 
			0,  /*January 1st=*/ 0, 31 * PERDAY, 59 * PERDAY, 90 * PERDAY, 
				/*May 1st=*/ 120 * PERDAY, 151 * PERDAY, 181 * PERDAY, 212 * PERDAY, 
				/*September 1st=*/ 243 * PERDAY, 273 * PERDAY, 304 * PERDAY, 334 * PERDAY
		};

		// # of seconds past in year, given start of a month 
		static const UInt32 g_tableSecondsPastInLeapYear[13] = { 
			0,  /*January 1st=*/ 0, 31 * PERDAY, 60 * PERDAY, 91 * PERDAY, 
				/*May 1st=*/ 121 * PERDAY, 152 * PERDAY, 182 * PERDAY, 213 * PERDAY, 
				/*September 1st=*/ 244 * PERDAY, 274 * PERDAY, 305 * PERDAY, 335 *PERDAY
		};

		// # of seconds in given month
		static const UInt32 g_tableSecondsInMonthLY[13] = {
			0,  /*January*/ 31 * PERDAY, 29 * PERDAY, 31 * PERDAY, 30 * PERDAY, 
				/*May=*/ 31 * PERDAY, 30 * PERDAY, 31 * PERDAY, 31 * PERDAY, 
				/*September=*/ 30 * PERDAY, 31 * PERDAY, 30 * PERDAY, 31 * PERDAY
		};

		// # of seconds in given month
		static const UInt32 g_tableSecondsInMonthNLY[13] = {
			0,  /*January*/ 31 * PERDAY, 28 * PERDAY, 31 * PERDAY, 30 * PERDAY, 
				/*May=*/ 31 * PERDAY, 30 * PERDAY, 31 * PERDAY, 31 * PERDAY, 
				/*September=*/ 30 * PERDAY, 31 * PERDAY, 30 * PERDAY, 31 * PERDAY
		};

		#undef PERDAY
	}
};

#endif	// __TimeConstants_h__

//	End of TimeConstants.h


