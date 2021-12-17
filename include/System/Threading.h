/////////
//  Threading.h
//  Copyright (C) 2017 by Wiley Black
/////////

#ifndef __WBThreading_h__
#define __WBThreading_h__

#include <mutex>

#include "../wbFoundation.h"

#undef Yield

#ifndef _WINDOWS

	#if defined(GCC_VERSION) && (GCC_VERSION <= 40303)
		#include <sched.h>
	#elif defined(GCC_VERSION) && (GCC_VERSION > 40303)
		#include <thread>
	#endif	
	#include <pthread.h>
	#include <signal.h>

#endif

namespace wb { namespace sys { namespace threading {

	class Thread
	{
	public:

		typedef UInt32(*ThreadStartRoutine) (void *);		

	private:		

		#ifdef _WINDOWS
		HANDLE m_hThread;
		DWORD m_ThreadId;				
		#else
		pthread_t m_Thread;
		bool	  m_bAlive;
		UInt32	  m_ExitCode;

		volatile sig_atomic_t m_Start;
		volatile sig_atomic_t m_Dying;

		static void OnThreadExiting(void* pThread) {
			// On Linux, this routine will be called at the exit of a thread.  We use it to set
			// the 'dying' flag that will indicate to IsAlive() that the thread is dead or very
			// nearly so.
			((Thread*)pThread)->m_Dying = 1;
		}
		#endif

		// To facilitate having a different function prototype between Windows and Linux, we
		// insert a little launcher stub that unwraps a little data structure we allocated 
		// and calls the standardized ThreadStartRoutine prototype.  In Linux, we also need
		// to implement a facility for non-blocking polling of completion, which we setup here.
		struct ThreadLaunchData { ThreadStartRoutine pStart; void* pParam; Thread* pThread; };
		#ifdef _WINDOWS
		static DWORD WINAPI ThreadLauncher(void* pTLD) { 
		#else
		static void* ThreadLauncher(void* pTLD) {			
		#endif
			ThreadStartRoutine pStart = ((ThreadLaunchData*)pTLD)->pStart;
			void* pParam = ((ThreadLaunchData*)pTLD)->pParam;			
			#ifdef _WINDOWS
			delete (ThreadLaunchData*)pTLD;
			return (DWORD)(*pStart)(pParam);
			#else
			Thread* pThread = ((ThreadLaunchData*)pTLD)->pThread;
			delete (ThreadLaunchData*)pTLD;
			pThread->m_Dying = 0;
			UInt32 ret;

			// According to 24.4.7.2 Atomic Types: You can assume that pointer types are atomic.  I'm
			// not certain that we are preventing relocation of addresses here, but it seems we'd
			// be having a hard time regardless if objects can move around while we have active pointers
			// to them.
			pthread_cleanup_push(OnThreadExiting, (void*)pThread);
			
			// Wait for start indicator
			while (pThread->m_Start == 0) Yield();

			ret = (*pStart)(pParam);
			// We may or may not exit through this route (pStart may not return), but if we do then it's
			// time to invoke the cleanup handler.
			pthread_cleanup_pop(1);
			return (void*)(UInt64)ret;
			#endif
		}

	public:								

		/// <param name="pFunction">Pointer to a function of the form:
		/// UInt32 ThreadProc(void* lpParameter);
		/// </param>
		Thread(ThreadStartRoutine pFunction, void* pParameter)
		{
			ThreadLaunchData* pTLD = new ThreadLaunchData;
			pTLD->pStart = pFunction;
			pTLD->pParam = pParameter;
			pTLD->pThread = this;			

			#ifdef _WINDOWS
			m_hThread = nullptr;			
			m_ThreadId = 0;

			m_hThread = ::CreateThread(
				/*lpThreadAttributes=*/ nullptr,
				/*dwStackSize=*/ 0,
				/*lpStartAddress=*/ &ThreadLauncher,
				/*lpParameter=*/ pTLD,
				/*dwCreationFlags=*/ CREATE_SUSPENDED,
				&m_ThreadId
				);
			if (m_hThread == nullptr) {
				delete pTLD;
				Exception::ThrowFromWin32(::GetLastError());
			}
			#else
			m_Start = 0;
			m_bAlive = false;

			int errstatus = pthread_create(&m_Thread, nullptr, &ThreadLauncher, pTLD);
			if (errstatus != 0)
			{
				delete pTLD;
				Exception::ThrowFromErrno(errstatus);
			}
			m_bAlive = true;
			#endif
		}

		~Thread()
		{
			#ifdef _WINDOWS
			if (m_hThread != nullptr) {
				Join();
				::CloseHandle(m_hThread);
				m_hThread = nullptr;
			}
			#else
			if (m_bAlive) {
				Join();
				m_bAlive = false;
			}
			#endif
		}

		void Start()
		{
			#ifdef _WINDOWS
			if (::ResumeThread(m_hThread) == (DWORD)-1) Exception::ThrowFromWin32(::GetLastError());
			#else
			m_Start = 1;
			#endif
		}

		void Join()
		{
			#ifdef _WINDOWS
			if (::WaitForSingleObject(m_hThread, INFINITE) != WAIT_OBJECT_0) 
				throw Exception("Error querying thread completion status.");
			#else
			if (m_bAlive) {
				void* retval;
				int errstatus = pthread_join(m_Thread, &retval);
				if (errstatus != 0) Exception::ThrowFromErrno(errstatus);
				m_ExitCode = (UInt32)(UInt64)retval;
				m_bAlive = false;
			}
			#endif
		}

		bool IsAlive()
		{			
			#ifdef _WINDOWS
			DWORD dwResult = ::WaitForSingleObject(m_hThread, 0);
			if (dwResult == WAIT_OBJECT_0) return false;
			if (dwResult == WAIT_TIMEOUT) return true;
			throw Exception("Error querying thread status.");
			#else
			if (!m_bAlive) return false;
			if (m_Dying == 0) return true;
			// If m_Dying is 1 then we have already called the cleanup handler.  We can't
			// retrieve the exit code yet though, so we perform a Join().  This isn't exactly
			// what the user asked for, but the Join() time should be short because we are
			// already in the cleanup phase, and this way the exit code is safely available
			// by the time we return false from IsAlive().
			Join();
			return true;
			#endif
		}		

		UInt32	GetExitCode()
		{
			#ifdef _WINDOWS
			DWORD ret;
			if (!::GetExitCodeThread(m_hThread, &ret)) Exception::ThrowFromWin32(::GetLastError());
			return (UInt32)ret;
			#else
			assert (!m_bAlive);
			return m_ExitCode;
			#endif
		}

		static void Yield()
		{
			#ifdef _WINDOWS
			Sleep(0);
			#elif defined(GCC_VERSION) && (GCC_VERSION > 40303)
				/** Not sure exactly which GCC Version will provide this support. **/
			std::this_thread::yield();
			#else
			sched_yield();
			#endif
		}

		static void Sleep(Int32 Milliseconds)
		{
			#ifdef _WINDOWS
			::Sleep(Milliseconds);				
			#else
			timespec req, rem;
			req.tv_sec = Milliseconds / 1000;
			req.tv_nsec = (Milliseconds % 1000l) * 1000l;
			if (nanosleep(&req, &rem) != 0) Exception::ThrowFromErrno(errno);					
			#endif
		}
	};	

	enum class EventResetMode
	{
		/// <summary>
		/// When signaled, the EventWaitHandle resets automatically after releasing a single thread. If no threads are waiting, the 
		/// EventWaitHandle remains signaled until a thread blocks, and resets after releasing the thread.
		/// </summary>
		AutoReset,

		/// <summary>
		/// When signaled, the EventWaitHandle releases all waiting threads and remains signaled until it is manually reset.
		/// </summary>
		ManualReset
	};

	// The EventWaitHandle class is based in part on the pevents library / WIN32 Events for POSIX from NeoSmart Technologies,
	// released under the terms of the MIT License.  This class could be considered a fork and certainly some concepts can be 
	// attributed to their design, and as such their original code remains:
	//		Copyright (C) 2011 - 2019 by NeoSmart Technologies
	// Their code is described here:
	//		https://neosmart.net/blog/2011/waitformultipleobjects-and-win32-events-for-linux-and-read-write-locks-for-windows/
	// And provided here:
	//		https://github.com/NeoSmart/PEvents
	class EventWaitHandle
	{			
	private:

		#ifdef _WINDOWS
		HANDLE m_hEvent;		
		#else
		#error Warning: implementation written but not compiled or tested yet.

		bool				m_bConditionValid;
		bool				m_bMutexValid;
		pthread_cond_t		m_Condition;
		pthread_mutex_t		m_Mutex;
		EventResetMode		m_Mode;
		std::atomic<bool>	m_State;

		bool UnlockedWaitForEvent(UInt32 TimeoutInMilliseconds) 
		{
			bool ret = false;
			int code;

			// memory_order_relaxed: `State` is only set to true with the mutex held, and we require
			// that this function only be called after the mutex is obtained.
			if (!m_State.load(std::memory_order_relaxed)) {
				// Zero-timeout event state check optimization
				if (TimeoutInMilliseconds == 0) return false;

				timespec ts;
				if (TimeoutInMilliseconds != INFINITE) {
					timeval tv;
					gettimeofday(&tv, NULL);

					uint64_t nanoseconds = ((uint64_t)tv.tv_sec) * 1000 * 1000 * 1000 +
						milliseconds * 1000 * 1000 + ((uint64_t)tv.tv_usec) * 1000;

					ts.tv_sec = (time_t)(nanoseconds / 1000 / 1000 / 1000);
					ts.tv_nsec = (long)(nanoseconds - ((uint64_t)ts.tv_sec) * 1000 * 1000 * 1000);
				}

				for (;;)
				{
					// Regardless of whether it's an auto-reset or manual-reset event:
					// wait to obtain the event, then lock anyone else out
					if (TimeoutInMilliseconds != INFINITE) {
						code = pthread_cond_timedwait(&m_Condition, &m_Mutex, &ts);
						if (code == ETIMEDOUT) return false;
					}
					else {
						code = pthread_cond_wait(&m_Condition, &m_Mutex);
					}
					if (code != 0) Exception::ThrowFromErrno(code);

					// memory_order_relaxed: ordering is guaranteed by the mutex, as `State = true` is
					// only ever written with the mutex held.

					// We reach this check if pthread_cond_xxx() returns 0, which could happen when the
					// state is not/no longer signaled.  If the state is not signaled, we just go back to waiting
					// more.					
					if (!m_State.load(std::memory_order_relaxed)) {
						ret = true; 
						break;
					}
				}
			}
			else if (m_Mode == EventResetMode::AutoReset) {
				// It's an auto-reset event that's currently available;
				// we need to stop anyone else from using it
				ret = true;
			}
			else {
				// We're trying to obtain a manual reset event with a signaled state; don't do anything
			}

			if (ret && m_Mode == EventResetMode::AutoReset) {
				// We've only accquired the event if the wait succeeded
				// memory_order_relaxed: we never act on `State == true` without fully synchronizing
				// or grabbing the mutex, so it's OK to use relaxed semantics here.
				m_State.store(false, std::memory_order_relaxed);
			}

			return ret;
		}
		#endif		

		void Create(bool initialState, EventResetMode mode, const wchar_t* pszName = nullptr)
		{			
			if (mode != EventResetMode::AutoReset && mode != EventResetMode::ManualReset) throw ArgumentException("Unrecognized EventResetMode value.");
			#ifdef _WINDOWS						
			m_hEvent = ::CreateEvent(nullptr, (mode == EventResetMode::ManualReset), initialState, pszName);
			if (m_hEvent == nullptr) Exception::ThrowFromWin32(::GetLastError());
			#else
			m_bConditionValid = m_bMutexValid = false;
			int code;
			code = pthread_cond_init(&m_Condition, 0);
			if (code != 0) Exception::ThrowFromErrno(code);
			m_bConditionValid = true;
			code = pthread_mutex_init(&m_Mutex, 0);
			if (code != 0) Exception::ThrowFromErrno(code);
			m_bMutexValid = true;
			m_Mode = mode;
			m_State.store(initialState, std::memory_order_release);
			#endif
		}

	public:				

		#ifdef _WINDOWS
		EventWaitHandle(bool initialState, EventResetMode mode, const osstring& name)
		{
			if (name.length() > MAX_PATH) throw ArgumentException("EventWaitHandle name cannot exceed MAX_PATH length.");
			Create(initialState, mode, name.c_str());
		}
		#else
		EventWaitHandle(bool initialState, EventResetMode mode, const osstring& name)
		{
			Create(initialState, mode, nullptr);
		}
		#endif

		EventWaitHandle(bool initialState, EventResetMode mode)
		{
			Create(initialState, mode, nullptr);
		}

		~EventWaitHandle()
		{
			#ifdef _WINDOWS
			if (m_hEvent != nullptr) {
				::CloseHandle(m_hEvent);
				m_hEvent = nullptr;
			}
			#else
			int code;
			if (m_bConditionValid)
			{
				m_bConditionValid = false;
				code = pthread_cond_destroy(&m_Condition);
				if (code != 0) Exception::ThrowFromErrno(code);
			}
			if (m_bMutexValid)
			{
				m_bMutexValid = false;
				code = pthread_mutex_destroy(&m_Mutex);
				if (code != 0) Exception::ThrowFromErrno(code);
			}
			#endif
		}
		
		/// <summary>
		/// Sets the state of the event to signaled, allowing one or more waiting threads to proceed.
		/// </summary>
		void Set()
		{
			#ifdef _WINDOWS
			if (!::SetEvent(m_hEvent)) Exception::ThrowFromWin32(::GetLastError());
			#else
			int code;
			code = pthread_mutex_lock(&m_Mutex);
			if (code != 0) Exception::ThrowFromErrno(code);

			if (m_Mode == EventResetMode::ManualReset)
			{
				// According to the Win32 API, for a manual reset event, any number of waiting threads (or threads
				// that subsequently begin wait operations for the event) can be released while the object's state
				// is signaled.

				m_State.store(true, std::memory_order_release);

				code = pthread_mutex_unlock(&m_Mutex);
				if (code != 0) Exception::ThrowFromErrno(code);

				// Unblock all threads waiting on this condition.
				code = pthread_cond_broadcast(&m_Condition);
				if (code != 0) Exception::ThrowFromErrno(code);
			}
			else
			{
				// According to the Win32 API, for an auto-reset event, only a single waiting thread is released
				// at which time the system automatically sets the state to nonsignaled.  If no threads are waiting,
				// the state remains unchanged.

				m_State.store(true, std::memory_order_release);

				code = pthread_mutex_unlock(&m_Mutex);
				if (code != 0) Exception::ThrowFromErrno(code);

				code = pthread_cond_signal(&m_Condition);
				if (code != 0) Exception::ThrowFromErrno(code);
			}
			#endif
		}

		/// <summary>
		/// Sets the state of the event to nonsignaled, causing threads to block.
		/// </summary>
		void Reset()
		{
			#ifdef _WINDOWS
			if (!::ResetEvent(m_hEvent)) Exception::ThrowFromWin32(::GetLastError());
			#else
			// Only the sequencing of concurrent SetEvent() calls is well defined.			
			m_State.store(false, std::memory_order_relaxed);
			#endif
		}

		/// <summary>
		/// Blocks the current thread until this event receives a signal.
		/// </summary>
		/// <param name="TimeoutInMilliseconds">Optional timeout to wait.</param>
		/// <returns>True if the event is signaled, false otherwise.</returns>
		bool WaitOne(UInt32 TimeoutInMilliseconds = INFINITE)
		{
			#ifdef _WINDOWS
			DWORD state = ::WaitForSingleObject(m_hEvent, TimeoutInMilliseconds);
			switch (state)
			{
			case WAIT_ABANDONED: throw Exception("Abandoned mutex.");		// In C#, AbandonedMutexException
			case WAIT_OBJECT_0: return true;
			case WAIT_TIMEOUT: return false;
			case WAIT_FAILED: Exception::ThrowFromWin32(::GetLastError());
			default: throw NotImplementedException("Unrecognized WaitForSingleObject response.");
			}
			#else
			int code;

			// Optimization: bypass acquiring the event lock if the state atomic is unavailable.
			// memory_order_relaxed: This is just an optimization, it's OK to be biased towards a stale
			// value here, and preferable to synchronizing CPU caches to get a more accurate result.
			if (TimeoutInMilliseconds == 0 && !m_State.load(std::memory_order_relaxed)) return false;

			// Optimization: early return in case of success for manual reset events only.
			if (m_Mode == EventResetMode::ManualReset && m_State.load(std::memory_order_relaxed)) {
				// A memory barrier is required here. This is still cheaper than a syscall.
				// See https://github.com/neosmart/pevents/issues/18
				if (m_State.load(std::memory_order_acquire)) return true;
			}

			code = pthread_mutex_lock(&m_Mutex);
			if (code != 0) Exception::ThrowFromErrno(code);
			
			bool ret = false;
			try
			{
				ret = UnlockedWaitForEvent(TimeoutInMilliseconds);
			}
			catch (...)
			{
				pthread_mutex_unlock(&m_Mutex);
				throw;
			}

			code = pthread_mutex_unlock(&m_Mutex);
			if (code != 0) Exception::ThrowFromErrno(code);

			return ret;
			#endif
		}
	};

} } }// End namespace

#endif  // __WBThreading_h__

//  End of Threading.h

