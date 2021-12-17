/////////
//	Monitor.h
////

#ifndef __WBMonitor_h__
#define __WBMonitor_h__

// Enable this to track the current owner of a lock.
//#define LOCK_DEBUGGING

#include "../wbFoundation.h"

namespace wb { namespace sys { namespace threading {

	class ILockable
	{		
	protected:

		friend class Monitor;
		virtual void Lock_Enter(const std::string& source_label = "") = 0;
		virtual bool Lock_TryEnter(const std::string& source_label = "") = 0;
		virtual void Lock_Exit() = 0;

	public:
	};	

	class Monitor 
	{
	public:
		static void Enter(ILockable& obj, const std::string& source_label = "") { obj.Lock_Enter(source_label); }
		static bool TryEnter(ILockable& obj, const std::string& source_label = "") { return obj.Lock_TryEnter(source_label); }
		static void Exit(ILockable& obj) { obj.Lock_Exit(); }
	};

} } }//	End namespace

namespace wb
{
	#ifdef _WINDOWS

	class CriticalSection : public wb::sys::threading::ILockable
	{
		// On a Windows 8.1 PRO x64 system, CRITICAL_SECTION object is 40 bytes
		// and requires 400ns to initialize.
		CRITICAL_SECTION	m_CS;

		// TODO: Consider using std::recursive_mutex as a core object in order to achieve cross-platform support.
		// If I go this route, should make CriticalSection's base class the std::recursive_mutex so that it is
		// interchangable.  And should consider making Monitor, Lock, and TryLock accept std::recursive_mutex.
		// Though should consider whether this constrains debugging of deadlocks with special tracking code.

	protected:

		// Windows SDK documentation indicates it is fine for the same thread to nest calls
		// to Enter or TryEnter as long as there are matching Exit calls.
		void Lock_Enter(const std::string& source_label = "") override { 
			::EnterCriticalSection(&m_CS); 
			#ifdef LOCK_DEBUGGING
			owner = source_label; 
			#endif
		}
		bool Lock_TryEnter(const std::string& source_label = "") override { 
			#ifdef LOCK_DEBUGGING
			if (::TryEnterCriticalSection(&m_CS) != 0)
			{
				owner = source_label;
				return true;
			}
			return false;
			#else
			return (::TryEnterCriticalSection(&m_CS) != 0);
			#endif
		}
		void Lock_Exit() override { ::LeaveCriticalSection(&m_CS); }

	public:

		#ifdef LOCK_DEBUGGING
		std::string owner;
		#endif

		CriticalSection()
		{
			::InitializeCriticalSection(&m_CS);
		}

		~CriticalSection()
		{
			::DeleteCriticalSection(&m_CS);
		}		

		void Enter(const std::string& source_label = "") { Lock_Enter(source_label); }
		bool TryEnter(const std::string& source_label = "") { return Lock_TryEnter(source_label); }
		void Exit() { Lock_Exit(); }

	};	

	#endif

	/// <summary>Lock provides C++ RAII access to a critical section.  The caller will pass a reference to the critical
	/// section/mutex object to be locked to the Lock constructor.  If another thread owns the lock the constructor will
	/// block until the lock is acquired.  The lock is released at destruction of the Lock object.</summary>
	class Lock
	{
		wb::sys::threading::ILockable&	m_rTarget;
	public:
		Lock(wb::sys::threading::ILockable& obj, const std::string& source_label = "") : m_rTarget(obj) { wb::sys::threading::Monitor::Enter(m_rTarget, source_label); }
		~Lock() { wb::sys::threading::Monitor::Exit(m_rTarget); }
	};

	/// <summary>TryLock is similar to Lock and provides C++ RAII access to a critical section but without blocking.  The
	/// caller will initialize the TryLock by providing the critical section/mutex object to the TryLock constructor.  Unlike
	/// Lock, the TryLock constructor will return immediately.  The caller then checks IsLocked() to determine if the lock
	/// was actually acquired.  If the lock was acquired, then destruction of the TryLock will release it.  If the lock was
	/// not acquired, destruction of the TryLock has no effect.</summary>
	class TryLock
	{
		wb::sys::threading::ILockable&	m_rTarget;
		bool m_Acquired;
	public:
		TryLock(wb::sys::threading::ILockable& obj, const std::string& source_label = "") : m_rTarget(obj) { m_Acquired = wb::sys::threading::Monitor::TryEnter(m_rTarget, source_label); }
		bool IsLocked() { return m_Acquired; }
		~TryLock() { if (m_Acquired) { wb::sys::threading::Monitor::Exit(m_rTarget); m_Acquired = false; } }
	};
};

#endif	// __WBMonitor_h__

//	End of Monitor.h

