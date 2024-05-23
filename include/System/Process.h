/////////
//  Process.h
//  Copyright (C) 2017 by Wiley Black
/////////

#ifndef __WBProcess_h__
#define __WBProcess_h__

#include "../wbFoundation.h"
#include "Threading.h"
#include "../IO/PipeStream.h"

#ifdef _WINDOWS

namespace wb { namespace sys {

	class Process
	{
		HANDLE m_hThread;
		HANDLE m_hProcess;

		string EnvironmentToStrings(const vector<string>& Environ)
		{
			string ret;

			for (size_t ii=0; ii < Environ.size(); ii++)
			{
				string entry(Environ[ii].c_str(), Environ[ii].length() + 1);
				ret.append(entry);
			}

			string final_terminator("\0", 1);
			ret.append(final_terminator);
			return ret;
		}

	public:
		Process() { m_hProcess = m_hThread = INVALID_HANDLE_VALUE; }
		~Process() { 
			if (m_hThread != INVALID_HANDLE_VALUE) {
				CloseHandle(m_hThread);
				m_hThread = INVALID_HANDLE_VALUE;
			}

			if (m_hProcess != INVALID_HANDLE_VALUE) {
				CloseHandle(m_hProcess);
				m_hProcess = INVALID_HANDLE_VALUE;
			}
		}

		// CommandLine should include the module name as well as any additional parameters.
		void Start(string FileName, string CommandLine)
		{
			::STARTUPINFO si;
			ZeroMemory(&si, sizeof(si));
			si.cb = sizeof(si);
			::PROCESS_INFORMATION pi;
			ZeroMemory(&pi, sizeof(pi));
			osstring osCommandLine = to_osstring(CommandLine);

			if (!::CreateProcess(
				to_osstring(FileName).c_str(),
				(TCHAR *)osCommandLine.c_str(),
				/*lpProcessAttributes=*/ nullptr,
				/*lpThreadAttributes=*/ nullptr,
				/*bInheritHandles=*/ true,
				/*dwCreationFlags=*/ 0,  
				/*lpEnvironment=*/ nullptr,			// When null, uses the environment of the parent process.
				/*lpCurrentDirectory=*/ nullptr,
				/*lpStartupInfo=*/ &si,
				/*lpProcessInformation=*/ &pi)
				) Exception::ThrowFromWin32(GetLastError());
			m_hThread = pi.hThread;
			m_hProcess = pi.hProcess;
		}

		// CommandLine should include the module name as well as any additional parameters.
		void Start(string FileName, string CommandLine, const vector<string>& Environment)
		{
			::STARTUPINFO si;
			ZeroMemory(&si, sizeof(si));
			si.cb = sizeof(si);
			::PROCESS_INFORMATION pi;
			ZeroMemory(&pi, sizeof(pi));
			osstring osCommandLine = to_osstring(CommandLine);

			string strEnviron = EnvironmentToStrings(Environment);

			// To pass UNICODE environment in, would need to add CREATE_UNICODE_ENVIRONMENT and use osstring.

			if (!::CreateProcess(
				to_osstring(FileName).c_str(),
				(TCHAR *)osCommandLine.c_str(),
				/*lpProcessAttributes=*/ nullptr,
				/*lpThreadAttributes=*/ nullptr,
				/*bInheritHandles=*/ true,
				/*dwCreationFlags=*/ 0,  
				/*lpEnvironment=*/ (void*)strEnviron.data(),
				/*lpCurrentDirectory=*/ nullptr,
				/*lpStartupInfo=*/ &si,
				/*lpProcessInformation=*/ &pi)
				) Exception::ThrowFromWin32(GetLastError());
			m_hThread = pi.hThread;
			m_hProcess = pi.hProcess;
		}

		// CommandLine should include the module name as well as any additional parameters.
		void Start(string FileName, string CommandLine, string WorkingDirectory, const vector<string>& Environment)
		{
			::STARTUPINFO si;
			ZeroMemory(&si, sizeof(si));
			si.cb = sizeof(si);
			::PROCESS_INFORMATION pi;
			ZeroMemory(&pi, sizeof(pi));
			osstring osCommandLine = to_osstring(CommandLine);

			string strEnviron = EnvironmentToStrings(Environment);

			// To pass UNICODE environment in, would need to add CREATE_UNICODE_ENVIRONMENT and use osstring.

			if (!::CreateProcess(
				to_osstring(FileName).c_str(),
				(TCHAR *)osCommandLine.c_str(),			// Important note: In UNICODE, this string may be modified.
				/*lpProcessAttributes=*/ nullptr,
				/*lpThreadAttributes=*/ nullptr,
				/*bInheritHandles=*/ true,
				/*dwCreationFlags=*/ 0,  
				/*lpEnvironment=*/ (void*)strEnviron.data(),
				/*lpCurrentDirectory=*/ to_osstring(WorkingDirectory).c_str(),
				/*lpStartupInfo=*/ &si,
				/*lpProcessInformation=*/ &pi)
				) Exception::ThrowFromWin32(GetLastError());
			m_hThread = pi.hThread;
			m_hProcess = pi.hProcess;
		}

		// CommandLine should include the module name as well as any additional parameters.
		void Start(string FileName, string CommandLine, wb::io::PipeStream& StdIn, wb::io::PipeStream& StdOut, wb::io::PipeStream& StdError, string WorkingDirectory, const vector<string>& Environment)
		{
			::STARTUPINFO si;
			ZeroMemory(&si, sizeof(si));
			si.cb = sizeof(si);
			si.hStdInput = StdIn.GetReadHandle();
			si.hStdOutput = StdOut.GetWriteHandle();
			si.hStdError = StdError.GetWriteHandle();
			si.dwFlags = STARTF_USESTDHANDLES;
			::PROCESS_INFORMATION pi;
			ZeroMemory(&pi, sizeof(pi));
			osstring osCommandLine = to_osstring(CommandLine);

			string strEnviron = EnvironmentToStrings(Environment);

			// To pass UNICODE environment in, would need to add CREATE_UNICODE_ENVIRONMENT and use osstring.

			if (!::CreateProcess(
				to_osstring(FileName).c_str(),
				(TCHAR *)osCommandLine.c_str(),			// Important note: In UNICODE, this string may be modified.
				/*lpProcessAttributes=*/ nullptr,
				/*lpThreadAttributes=*/ nullptr,
				/*bInheritHandles=*/ true,
				/*dwCreationFlags=*/ 0,  
				/*lpEnvironment=*/ (void*)strEnviron.data(),
				/*lpCurrentDirectory=*/ to_osstring(WorkingDirectory).c_str(),
				/*lpStartupInfo=*/ &si,
				/*lpProcessInformation=*/ &pi)
				) Exception::ThrowFromWin32(GetLastError());
			m_hThread = pi.hThread;
			m_hProcess = pi.hProcess;
		}

		void StartShell(string Command)
		{
			Start("CMD.EXE", ("CMD.EXE /C " + Command).c_str());
		}

		bool HasExited()
		{
			DWORD dwResult = ::WaitForSingleObject(m_hProcess, 0);
			if (dwResult == WAIT_OBJECT_0) return true;
			if (dwResult == WAIT_TIMEOUT) return false;
			throw Exception("Error querying child process status.");
		}

		void WaitForExit()
		{
			if (::WaitForSingleObject(m_hProcess, INFINITE) != WAIT_OBJECT_0) 
				throw Exception("Error querying child process status.");
		}

		int GetExitCode()
		{
			if (!HasExited()) throw Exception("Cannot query exit code before process has exited.");
			DWORD dwExitCode;
			if (!::GetExitCodeProcess(m_hProcess, &dwExitCode)) Exception::ThrowFromWin32(::GetLastError());			
			return (int)dwExitCode;
		}
	};

} }// End namespace

#else		// Linux
#include <signal.h>
#include <stddef.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
//#include <libc-lock.h>
//#include <sysdep-cancel.h>

namespace wb { namespace sys {

	class Process
	{
		struct ChangeSignalAction
		{
			struct sigaction Previous;
			bool Modified;
			int m_signum;
			ChangeSignalAction(int signum, const struct sigaction* pNewAction)
			{
				m_signum = signum;
				if (sigaction (signum, pNewAction, &Previous) < 0) Exception::ThrowFromErrno(errno);
				Modified = true;
			}
			void Undo() 
			{
				if (Modified) {
					sigaction (m_signum, &Previous, (struct sigaction *) nullptr);
					Modified = false;
				}
			}
			~ChangeSignalAction() { Undo(); }
		};
		struct AddBlockSignal
		{
			sigset_t Previous;
			bool Modified;
			AddBlockSignal(const sigset_t* pAddSet)
			{
				if (sigprocmask (SIG_BLOCK, pAddSet, &Previous) < 0)
				{
					if (errno == ENOSYS) return;		// Not implemented.  I assume this means it's already blocked because it doesn't exist.
					Exception::ThrowFromErrno(errno);
				}
				Modified = true;
			}
			void Undo()
			{
				if (Modified) {
					sigprocmask (SIG_SETMASK, &Previous, (sigset_t *) nullptr);
					Modified = false;
				}
			}
			~AddBlockSignal() {	Undo(); }
		};

		pid_t	m_pid;
		int		m_ExitCode;
		bool	m_ChildRunning;

		void Start(const char* pszCommand, const char*const * ppArguments) 
		{
			struct sigaction sa;			

			/** I'm not exactly sure that we want to be blocking/ignoring the SIGINT and SIGQUIT signals.  I have simply copied the conceptual flow of the system()
				call.  That may not be the same desirement when we are doing this non-blocking Start() call. **/

			sa.sa_handler = SIG_IGN;
			sa.sa_flags = 0;
			sigemptyset (&sa.sa_mask);

			// TODO: If one of the ChangeSignalActions or AddBlockSignals is Undo()'ing (destructor) we need to be in a DO_LOCK() block according to the orignal source.
			// That gets muddy with exceptions...we might be processing an exception outside the DO_LOCK.  In any case, I'm not implementing the DO_LOCK() blocks at all
			// yet, so this function may not be thread-safe.  Or maybe it is, the original source is from inside libc I think, the rules may be different outside of it.

			// DO_LOCK ();
			//if (ADD_REF () == 0)
			//{			
			ChangeSignalAction ChangeInt(SIGINT, &sa);
			ChangeSignalAction ChangeQuit(SIGQUIT, &sa);
			//}
			//DO_UNLOCK ();
			
			sigaddset (&sa.sa_mask, SIGCHLD);
			AddBlockSignal AddBlock(&sa.sa_mask);

			pid_t pid = fork ();
			if (pid < (pid_t)0) Exception::ThrowFromErrno(errno);
			if (pid == (pid_t) 0)
			{
				/* Child side.  This code is executing in the new child process. */				

				/* Restore the signals.  */
				ChangeInt.Undo();
				ChangeQuit.Undo();
				AddBlock.Undo();
				// INIT_LOCK ();
				/* Exec the shell.  The execve() call does not return (the current program in this process is replaced). */
				execve (pszCommand, (char *const *) ppArguments, environ);
				try { Exception::ThrowFromErrno(errno); }
				catch (std::exception& ex) 
				{
					#ifdef _DEBUG
					printf("Failed to launch child process: %s\n", ex.what());
					#endif
				}				
				exit (127);		// Should only execute if execve() failed.
			}
			/* Parent side code returns */
			m_pid = pid;
			m_ChildRunning = true;
		}

	public:

		string ShellPath;
		string ShellName;

		Process()
		{
			ShellPath = "/bin/sh";			// Default path of the shell.
			ShellName = "sh";				// Default name of the shell.
			m_pid = (pid_t) -1;
			m_ExitCode = 127;
			m_ChildRunning = false;
		}

		int GetExitCode() { return m_ExitCode; }				

		void Start(string FileName) 
		{
			const char *new_argv[2];
			new_argv[0] = FileName.c_str();			
			new_argv[1] = NULL;
			Start(FileName.c_str(), new_argv);
		}

		/// <summary>Similar to Start(), but launches the specified command-line via the current system shell.  (Extension to .NET API)</summary>
		void StartShell(string Command)
		{			
			const char *new_argv[4];
			new_argv[0] = ShellName.c_str();
			new_argv[1] = "-c";
			new_argv[2] = Command.c_str();
			new_argv[3] = NULL;
			Start(ShellPath.c_str(), new_argv);
		}

		bool HasExited()
		{
			if (m_pid < (pid_t)-1) throw Exception("No Process started.");
			if (!m_ChildRunning) return true;			

			/* Note the system() is a cancellation point.  But since we call
				waitpid() which itself is a cancellation point we do not
				have to do anything here.  */
			for (;;)
			{
				int status;
				pid_t waitval = waitpid(m_pid, &status, WNOHANG);
				if (waitval == -1)
				{
					if (errno == EINTR) continue;			// Retry indefinitely.
					Exception::ThrowFromErrno(errno);
				}
				if (waitval == m_pid) 
				{ 
					m_ChildRunning = false; 
					if (WIFEXITED(status)) m_ExitCode = WEXITSTATUS(status);
					else m_ExitCode = -100;
					return true; 
				}
				return false;
			}
		}

		void WaitForExit()
		{
			for (;;) { if (HasExited()) return; wb::sys::threading::Thread::Yield(); }
		}
	};

} }// End namespace

#endif	// _WINDOWS

#endif  // __WBProcess_h__

//  End of Process.h

