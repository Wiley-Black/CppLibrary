/**	Stopwatch.h	
**/
/////////

#include "../wbFoundation.h"			// Ensures Windows.h is included first.

#ifndef __Stopwatch_h__
#define __Stopwatch_h__

/** Configuration **/

#ifdef _WINDOWS
	/** If STOPWATCH_STATIC_INIT is defined, then Stopwatch will use a static initialization approach to measuring the CountsPerSecond/clock frequency.  This technique
		will read from a boolean "FirstTime" variable, and if it is false it will enter into a static initialization routine protected by critical section.  If Stopwatches
		are created from multiple DLLs, this initialization may occur once for each DLL once a Stopwatch requests it in that particular DLL.

		If STOPWATCH_STATIC_INIT is not defined, then Stopwatch will call QueryPerformanceFrequency() for each Stopwatch object constructed.  On 10-10-2018 this was profiled
		to take only 4ns, although the Internet suggests that on some hardware configurations this can take a couple of microseconds.  The call is always made before the
		profile operation starts, so this should not affect the time measured. 
		
		On platforms besides Windows, there is no need for static initialization and this is not defined. **/
//#define STOPWATCH_STATIC_INIT
#endif

#include <string>
#include "../DateTime/TimeSpan.h"
#include "../System/Monitor.h"

namespace wb
{
	class Stopwatch
	{
		Int64	StartTime;		
		
		/// <summary>Running indicates that StartTime has been captured and time is running relative to StartTime.</summary>
		bool	Running;

		/// <summary>The elapsed time should be calculated as the sum of difference from StopTime (or now) since StartTime plus any
		/// Accumulated counts.  Accumulated counts are stored whenever the timer is stopped and allow for restarting the timer.</summary>
		Int64	Accumulated;		

		#ifdef STOPWATCH_STATIC_INIT
		// Implementation note: this class can be used across DLL boundaries and/or libraries that are statically linked.  This can lead to
		// the possibility that different member calls will happen in different DLLs, and therefore the static variables will actually
		// exist in duplicate.  This isn't much of an issue because the initialization can just be performed twice, but it makes it
		// critical that any use of the static variables be preceded by a call to CheckInit() to ensure that the globals are initialized.
		static bool FirstTime;
		static bool IsHighResolution;
		static Int64 CountsPerSecond;
		static wb::CriticalSection InitializationCS;
		static void OnFirstUse();
		static Int64 CountsSinceEpoch();
		#else
		bool IsHighResolution;
		Int64 CountsPerSecond;
		void OnFirstUse();
		Int64 CountsSinceEpoch() const;
		#endif

		static void CheckInit();				

	public:		

		/** Stopwatch calls are not thread-safe if the Stopwatch is running.  Thread-safety rules here are:

				1.	Before the Stopwatch has started or after the Stopwatch is Stop()'d or Reset() then 
					it is safe and reliable to call the read-only functions GetElapsed...() or ToString() 
					on any thread.  It is important to be sure that the Stop() or Reset() call has 
					completed before other threads call the GetElapsed...() or ToString() methods.  This
					allows profiling in one thread and then the Stop()'d result to be passed around to
					other threads as a result as long as the profiling is ensured completed before review.

				2.	It is always safe to call Stopwatch functions from multiple threads as long as the
					Stopwatch object is constrained to a single thread.  For example, if you construct 
					Stopwatch objects A and B on threads #1 and #2 respectively then it is safe to make 
					any calls on A from thread #1 and calls on B from thread #2.  It could be unsafe to 
					make calls on stopwatch object A from thread #2 or object B from thread #1 in accordance
					with rule #1 above.
		**/

		/// <summary>Initializes a new Stopwatch.  The Stopwatch is stopped until Start() is called.</summary>
		/// <seealso>Static StartNew() function to initialize and immediately start a Stopwatch.</seealso>
		Stopwatch();

		/// <summary>Initializes a Stopwatch and immediately starts it.  Thread-safe.</summary>
		static Stopwatch StartNew();

		void Start();
		void Stop();
		void Restart();
		void Reset();

		TimeSpan	GetElapsed() const;
		double		GetElapsedMilliseconds() const;
		double		GetElapsedSeconds() const;

		//Int64		GetElapsedNanoseconds() const;

		std::string ToString(int Precision = 9) const;		
	};

	/** Non-class operations **/

	inline std::string to_string(const Stopwatch& Value) { return Value.ToString(); }

	/////////
	//  Implementation
	//

	#if defined(PrimaryModule) && defined(STOPWATCH_STATIC_INIT)
	/*static*/ bool Stopwatch::FirstTime = true;
	/*static*/ bool Stopwatch::IsHighResolution = false;
	/*static*/ Int64 Stopwatch::CountsPerSecond = 0;
	/*static*/ wb::CriticalSection Stopwatch::InitializationCS;	
	#endif

	inline /*static if STOPWATCH_STATIC_INIT*/ void Stopwatch::OnFirstUse()
	{
		#ifdef STOPWATCH_STATIC_INIT
		wb::Lock InitLock(InitializationCS);
		if (!FirstTime) return;		// Can happen if the mutex went up in a different thread and we got blocked before we got here.	
		#endif
	
		#ifdef _WINDOWS
		LARGE_INTEGER liCountsPerSecond;
		if (QueryPerformanceFrequency(&liCountsPerSecond) == 0) IsHighResolution = false;
		else
		{
			if (liCountsPerSecond.QuadPart != 0) 
			{ 
				IsHighResolution = true; 
				CountsPerSecond = liCountsPerSecond.QuadPart; 
			}
			else 
				IsHighResolution = false;
		}	
		if (!IsHighResolution) CountsPerSecond = 1000;	
		#else
		// Using usec units:	
		CountsPerSecond = 1000000;
		#endif
	
		#ifdef STOPWATCH_STATIC_INIT
		FirstTime = false;
		#endif
	}

	inline /*static*/ void Stopwatch::CheckInit() { 
		#ifdef STOPWATCH_STATIC_INIT
		if (FirstTime) OnFirstUse(); 
		#endif
		// When not using static init, OnFirstUse() is called by the constructor and is not static.
	}	

	inline /*static*/ Stopwatch Stopwatch::StartNew() { Stopwatch ret; ret.Start(); return ret; }
		
	inline Stopwatch::Stopwatch() { 
		#ifdef STOPWATCH_STATIC_INIT
		CheckInit(); 
		#else
		OnFirstUse();
		#endif
		Running = false; 
		Accumulated = 0; 
	}	

	inline /*static if STOPWATCH_STATIC_INIT*/ Int64 Stopwatch::CountsSinceEpoch() const
	{				
		#ifdef _WINDOWS					
		#ifdef STOPWATCH_STATIC_INIT
		assert (!FirstTime);
		#endif
		if (IsHighResolution)
		{
			LARGE_INTEGER now;
			if (!QueryPerformanceCounter(&now)) throw Exception("Unable to retrieve high-performance system counter despite capability being reported as available.");
			return now.QuadPart;
		}
		else
		{
			return (Int64)GetTickCount64();
		}
		#else
		// The tv_usec is the highest precision we're using here, so use units of usec.
        struct timeval tv;
        gettimeofday(&tv, NULL);
        return ((Int64)tv.tv_sec) * 1000000) + (Int64)tv.tv_usec;
		#endif
	}

	inline void Stopwatch::Start()
	{
		if (Running) return;
		Running = true;
		CheckInit();
		StartTime = CountsSinceEpoch();
	}	

	inline void Stopwatch::Stop()
	{
		if (Running)
		{
			CheckInit();
			Int64 StopTime = CountsSinceEpoch();
			Accumulated += (StopTime - StartTime);
		}
		Running = false;
	}

	inline void Stopwatch::Reset()
	{
		Running = false;
		Accumulated = 0;
	}

	inline void Stopwatch::Restart()
	{
		Reset();
		Start();
	}

#if 0	// GetElapsedNanoseconds() can rollover after something like 30 minutes... 
	inline Int64 Stopwatch::GetElapsedNanoseconds() const
	{
		CheckInit();

		// The Stopwatch may have been Start()-Stop()'d multiple times without being Reset().  Each time Stop() is called, the period is 
		// added to Accumulated, so Accumulated is what we want to represent as the overall elapsed time.  It's also possible that the
		// stopwatch is still running as GetElapsedNanoseconds() is called.  Stop() may have never been called.  We can handle this by
		// adding the current period on top of Accumulated.
		Int64 Elapsed = Accumulated;		
		if (Running) Elapsed += (CountsSinceEpoch() - StartTime);
		
		// i.e. (1 count * 1000000000 ns/s) / 100 counts/s = 10000000 ns
		// i.e. (1 count * 1000000000 ns/s) / 10000000000 counts/s = 0 ns (0.1 ns rounded down)		

		// Careful to multiply first to avoid losing precision.
		if (CountsPerSecond <= 0) throw NotSupportedException("Stopwatch initialization failed.");		// Prevent a divide by zero fault.
		return (Elapsed * 1000000000ll) / CountsPerSecond;
	}
#endif

	inline double Stopwatch::GetElapsedMilliseconds() const
	{
		CheckInit();

		// The Stopwatch may have been Start()-Stop()'d multiple times without being Reset().  Each time Stop() is called, the period is 
		// added to Accumulated, so Accumulated is what we want to represent as the overall elapsed time.  It's also possible that the
		// stopwatch is still running as GetElapsedNanoseconds() is called.  Stop() may have never been called.  We can handle this by
		// adding the current period on top of Accumulated.
		Int64 Elapsed = Accumulated;
		if (Running) Elapsed += (CountsSinceEpoch() - StartTime);		

		// Careful to multiply first to avoid losing precision.
		if (CountsPerSecond <= 0) throw NotSupportedException("Stopwatch initialization failed.");		// Prevent a divide by zero fault.
		return ((double)Elapsed * 1000.0) / (double)CountsPerSecond;
	}

	inline TimeSpan Stopwatch::GetElapsed() const {	return TimeSpan::FromSeconds(GetElapsedSeconds()); }
	inline double Stopwatch::GetElapsedSeconds() const { return (double)GetElapsedMilliseconds() / 1000.0; }

	inline std::string Stopwatch::ToString(int Precision /*= 9*/) const
	{
		return std::string("Stopwatch (") + GetElapsed().ToString(Precision) + ")";
	}
};

#endif	// __Stopwatch_h__

//	End of Stopwatch.h


