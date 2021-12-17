/*	MemoryInfo.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBMemoryInfo_h__
#define __WBMemoryInfo_h__

#include "../wbFoundation.h"

#ifdef _WINDOWS

/** Include the PSAPI for GetProcessMemoryInfo **/

#define PSAPI_VERSION 1								// For compatability with older Windows.
#include <psapi.h>
#pragma comment(lib, "psapi.lib")					// For GetProcessMemoryInfo() support.

#else
#include <unistd.h>
#include <ios>
#include <iostream>
#include <fstream>
#include <string>
#endif

namespace wb
{	
	namespace diagnostics
	{
		class Process
		{
		public:
			
			/// <summary>
			/// Retrieves the process' current memory usage.  RealMemory will provide the number of bytes of
			/// RAM in use by the calling process.  VirtualMemory will provide the number of bytes of virtual
			/// memory in use by the calling process.  Total memory usage for the process can be calculated
			/// as the sum of RealMemory and VirtualMemory.
			/// </summary>
			static void GetMemoryUsage(UInt64& RealMemory, UInt64& VirtualMemory);				

			/// <summary>Returns the number of bytes of RAM and virtual memory presently used by the calling process.</summary>
			static UInt64 GetMemoryUsage();			
		};
		
		/** Implementation - MemoryInfo **/

		#ifndef _WINDOWS		/** Linux solution **/
		
		/*static*/ void Process::GetMemoryUsage(UInt64& RealMemory, UInt64& VirtualMemory)
		{			
			using std::ios_base;
			using std::ifstream;
			using std::string;

			RealMemory = 0;
			VirtualMemory = 0;

			// 'file' stat seems to give the most reliable results
			ifstream stat_stream("/proc/self/stat", ios_base::in);

			// dummy vars for leading entries in stat that we don't care about
			string pid, comm, state, ppid, pgrp, session, tty_nr;
			string tpgid, flags, minflt, cminflt, majflt, cmajflt;
			string utime, stime, cutime, cstime, priority, nice;
			string O, itrealvalue, starttime;

			// the two fields we want
			UInt64 vsize;
			UInt64 rss;

			stat_stream >> pid >> comm >> state >> ppid >> pgrp >> session >> tty_nr
				>> tpgid >> flags >> minflt >> cminflt >> majflt >> cmajflt
				>> utime >> stime >> cutime >> cstime >> priority >> nice
				>> O >> itrealvalue >> starttime >> vsize >> rss; // don't care about the rest

			stat_stream.close();

			UInt64 page_size = sysconf(_SC_PAGE_SIZE); // in case x86-64 is configured to use 2MB pages

			VirtualMemory = vsize;
			RealMemory = rss * page_size;
		}		

		#else

		/** There appears to be a compiler bug where compilation of the GetMemoryUsage() member function
			from MemoryInfo.h or from wbFoundation.h will cause the following linker error:

			INK : fatal error C1905: Front end and back end not compatible (must target same processor).

			The workaround for now is to exclude this member function when compiling in x64 mode.  If
			the function is required, copy-and-paste it into a CPP module or use a newer compiler.
		**/

		inline /*static*/ void Process::GetMemoryUsage(UInt64& RealMemory, UInt64& VirtualMemory)
		{	
			::PROCESS_MEMORY_COUNTERS_EX pmc;
			pmc.cb = sizeof(pmc);
			
			if (!::GetProcessMemoryInfo(::GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc)))			
				Exception::ThrowFromWin32(::GetLastError());
			RealMemory = pmc.WorkingSetSize;
			VirtualMemory = pmc.PrivateUsage;
		}

		#endif		

		inline /*static*/ UInt64 Process::GetMemoryUsage()
		{
			UInt64 RealMemory, VirtualMemory;
			GetMemoryUsage(RealMemory, VirtualMemory);
			return RealMemory + VirtualMemory;
		}		
	}
}

#endif	// __WBMemoryInfo_h__

//	End of MemoryInfo.h


