/////////
//  Process.h
//  Copyright (C) 2014 by Wiley Black
/////////

#ifndef __WBProcess_h__
#define __WBProcess_h__

#include "wbFoundation.h"
#include "IO/Streams.h"
#include "IO/FileStream.h"

namespace wb
{
	namespace threading
	{
		#ifdef _WINDOWS			// Can be defined for Linux, just isn't yet.

		class Process
		{
		protected:

			/** Process monitoring & information **/
			PROCESS_INFORMATION ProcessInfo;

		public:

			Process()
			{
				ZeroMemory(&ProcessInfo, sizeof(ProcessInfo));
				ProcessInfo.hProcess = INVALID_HANDLE_VALUE;
				ProcessInfo.hThread = INVALID_HANDLE_VALUE;
			}

			~Process()
			{
				Close();
			}

			virtual void Close()
			{
				// Close process and thread handles. 
				if (ProcessInfo.hProcess != INVALID_HANDLE_VALUE)
				{
					::CloseHandle(ProcessInfo.hProcess);
					ProcessInfo.hProcess = INVALID_HANDLE_VALUE;
				}
				if (ProcessInfo.hThread != INVALID_HANDLE_VALUE)
				{
					::CloseHandle(ProcessInfo.hThread);
					ProcessInfo.hThread = INVALID_HANDLE_VALUE;
				}
			}

			virtual void WaitForExit()
			{
				if (ProcessInfo.hThread == INVALID_HANDLE_VALUE) throw Exception("Process must be started before WaitForExit can be called.");				
				if (::WaitForSingleObject(ProcessInfo.hProcess, INFINITE) == WAIT_FAILED) Exception::ThrowFromWin32(::GetLastError());					
			}

			bool HasExited()
			{
				if (ProcessInfo.hThread == INVALID_HANDLE_VALUE) throw Exception("Process must be started before HasExited can be called.");				
				DWORD Status = ::WaitForSingleObject(ProcessInfo.hProcess, 0);
				switch (Status)
				{
				case WAIT_ABANDONED: throw Exception("Abandoned process handle when checking exit status.");
				case WAIT_OBJECT_0: return true;
				case WAIT_TIMEOUT: return false;
				case WAIT_FAILED: Exception::ThrowFromWin32(::GetLastError());
				default: throw Exception("Unrecognized WaitForSingleObject state.");
				}
			}

			int GetExitCode()
			{
				if (ProcessInfo.hThread == INVALID_HANDLE_VALUE) throw Exception("Process must be started before GetExitCode can be called.");				
				DWORD Status = ::WaitForSingleObject(ProcessInfo.hProcess, 0);
				switch (Status)
				{
				case WAIT_ABANDONED: throw Exception("Abandoned process handle when checking exit status.");
				case WAIT_OBJECT_0: 
					{
						DWORD ExitCode;
						if (!::GetExitCodeProcess(ProcessInfo.hProcess, &ExitCode)) Exception::ThrowFromWin32(::GetLastError());
						return ExitCode;
					}
					return true;
				case WAIT_TIMEOUT: throw NotSupportedException("Process must have exited before exit code can be retrieved.");
				case WAIT_FAILED: Exception::ThrowFromWin32(::GetLastError());
				default: throw Exception("Unrecognized WaitForSingleObject state.");
				}
			}
			
			virtual void Start(string CommandLine, string WorkingDirectory = "")
			{				
				STARTUPINFO si;     				
				ZeroMemory( &si, sizeof(si) );
				si.cb = sizeof(si);

				osstring osCommandLine = to_osstring(CommandLine);
				vector<TCHAR> CommandLineArray(osCommandLine.begin(), osCommandLine.end());
				CommandLineArray.push_back('\0');
				TCHAR* pszCommandLine = &CommandLineArray[0];

				const TCHAR* pszWorkingDirectory = nullptr;
				if (WorkingDirectory.length() > 0) pszWorkingDirectory = to_osstring(WorkingDirectory).c_str();				

				BOOL Success = ::CreateProcess( 
					nullptr,		// module name, taken from first parameter of command line.
					pszCommandLine, // Command line including executable.  Quotes should enclose individual parameters where necessary.
					nullptr,        // Process handle not inheritable
					nullptr,        // Thread handle not inheritable
					false,	        // Set handle inheritance to FALSE
					0,              // No creation flags
					nullptr,        // Use parent's environment block
					pszWorkingDirectory,        // Use parent's starting directory 
					&si,            // Pointer to STARTUPINFO structure
					&ProcessInfo	// Pointer to PROCESS_INFORMATION structure
					);
				if (!Success) Exception::ThrowFromWin32(::GetLastError());				
			}
		};

		class RedirectedProcess : public Process
		{
		public:
			class PipeStream : public io::FileStream
			{
				friend class RedirectedProcess;

			public:
				static PipeStream FromHandle(HANDLE h, bool CanRead, bool CanWrite)
				{
					PipeStream ret;
					ret.m_Handle = h;
					ret.m_bCanRead = CanRead;
					ret.m_bCanWrite = CanWrite;
					return ret;
				}
				PipeStream() { }
				PipeStream(const PipeStream&) { throw NotSupportedException(); }
				PipeStream(PipeStream&& mv) { operator=(std::move(mv)); }
				PipeStream& operator=(const PipeStream&) { throw NotSupportedException(); }
				PipeStream& operator=(PipeStream&& mv) { io::FileStream::operator=(std::move(mv)); return *this; }
			
				bool IsDataAvailable()
				{
					DWORD dwAvailable;
					if (!::PeekNamedPipe(GetHandle(), nullptr, 0, nullptr, &dwAvailable, nullptr)) Exception::ThrowFromWin32(::GetLastError());
					return dwAvailable > 0;
				}				
			};

			PipeStream	m_ChildSTDIN_Rd;
			PipeStream	m_ChildSTDOUT_Wr;

		protected:
			PipeStream	m_ChildSTDIN_Wr;
			PipeStream	m_ChildSTDOUT_Rd;

		public:		

			RedirectedProcess()
			{
			}

			~RedirectedProcess()
			{				
			}

			PipeStream&	STDIN() { return m_ChildSTDIN_Wr; }
			PipeStream&	STDOUT() { return m_ChildSTDOUT_Rd; }

			// Starts a process with STDIN and STDOUT redirected to a pipe communicating with the parent (this) process.
			void Start(string CommandLine, string WorkingDirectory = "") override
			{				
				// Set the bInheritHandle flag so pipe handles are inherited. 
				SECURITY_ATTRIBUTES saAttr; 				 
				saAttr.nLength = sizeof(SECURITY_ATTRIBUTES); 
				saAttr.bInheritHandle = true; 
				saAttr.lpSecurityDescriptor = nullptr;

				/** Child STDIN and STDOUT handles **/
				HANDLE hChildStdIN_Rd;
				HANDLE hChildStdIN_Wr;
				HANDLE hChildStdOUT_Rd;
				HANDLE hChildStdOUT_Wr;

				// Create a pipe for the child's STDOUT
				if (!::CreatePipe(&hChildStdOUT_Rd, &hChildStdOUT_Wr, &saAttr, 0)) Exception::ThrowFromWin32(::GetLastError());
				// We only want the write handle for STDOUT to be inherited, not the read side...
				if (!::SetHandleInformation(hChildStdOUT_Rd, HANDLE_FLAG_INHERIT, 0)) Exception::ThrowFromWin32(::GetLastError());
				m_ChildSTDOUT_Rd = PipeStream::FromHandle(hChildStdOUT_Rd, true, false);
				m_ChildSTDOUT_Wr = PipeStream::FromHandle(hChildStdOUT_Wr, true, false);

				// Create a pipe for the child process's STDIN.  
				if (!::CreatePipe(&hChildStdIN_Rd, &hChildStdIN_Wr, &saAttr, 0)) Exception::ThrowFromWin32(::GetLastError());
				// We only want the read handle for STDIN to be inherited, not the write side...				 
				if (!::SetHandleInformation(hChildStdIN_Wr, HANDLE_FLAG_INHERIT, 0) ) Exception::ThrowFromWin32(::GetLastError());
				m_ChildSTDIN_Wr = PipeStream::FromHandle(hChildStdIN_Wr, false, true);
				m_ChildSTDIN_Rd = PipeStream::FromHandle(hChildStdIN_Rd, false, true);
				
				STARTUPINFO si;     				
				ZeroMemory( &si, sizeof(si) );
				si.cb = sizeof(si);
				si.hStdError = hChildStdOUT_Wr;
				si.hStdOutput = hChildStdOUT_Wr;
				si.hStdInput = hChildStdIN_Rd;
				si.dwFlags |= STARTF_USESTDHANDLES;				

				osstring osCommandLine = to_osstring(CommandLine);
				vector<TCHAR> CommandLineArray(osCommandLine.begin(), osCommandLine.end());
				CommandLineArray.push_back('\0');
				TCHAR* pszCommandLine = &CommandLineArray[0];

				const TCHAR* pszWorkingDirectory = nullptr;
				osstring osWorkingDirectory = to_osstring(WorkingDirectory);
				if (WorkingDirectory.length() > 0) pszWorkingDirectory = osWorkingDirectory.c_str();

				BOOL Success = ::CreateProcess( 
					nullptr,		// module name, taken from first parameter of command line.
					pszCommandLine, // Command line including executable.  Quotes should enclose individual parameters where necessary.
					nullptr,        // Process handle not inheritable
					nullptr,        // Thread handle not inheritable
					true,	        // Set handle inheritance to FALSE
					0,              // No creation flags
					nullptr,        // Use parent's environment block
					pszWorkingDirectory,        // Use parent's starting directory 
					&si,            // Pointer to STARTUPINFO structure
					&ProcessInfo	// Pointer to PROCESS_INFORMATION structure
					);				
				if (!Success) Exception::ThrowFromWin32(::GetLastError());				

				// We close the end of the pipes that was given to the child.  The parent no longer needs them - it only needs the other ends of
				// the pipe.  The pipes are not yet actually closed since the child has these handles as well, we're just closing our copy of
				// them.  This will, however, lead to the pipes closing asynchronously when the child exits ans the buffer has been completely
				// read.
				// Edit: Actually, avoid closing them.  The caller can use IsDataAvailable() to avoid blocking, and we'll close these pipes in
				// the RedirectedProcess destructor.  This too has a cavaet: The process won't register as closed until the pipes are completely
				// read out.
				//::CloseHandle(hChildStdOUT_Wr);
				//::CloseHandle(hChildStdIN_Rd);
			}

			void WaitForExit() override { throw NotSupportedException("Use WaitForExitAndRead() instead."); }
			
			template<typename string_type> void WaitForExitAndRead(string_type& ConsoleOutput)
			{
				ConsoleOutput.clear();
				byte buffer[4096];
				while (!HasExited())
				{
					if (STDOUT().IsDataAvailable())
					{
						Int64 nBytes = STDOUT().Read(buffer, sizeof(buffer));
						ConsoleOutput.append((string_type::const_pointer)buffer, (int)nBytes / sizeof(string_type::value_type));
					}
					else Thread::Yield();
				}				
				while (STDOUT().IsDataAvailable())
				{
					Int64 nBytes = STDOUT().Read(buffer, sizeof(buffer));
					ConsoleOutput.append((string_type::const_pointer)buffer, (int)nBytes / sizeof(string_type::value_type));
				}
			}

			void Close() override
			{
				Process::Close();
				m_ChildSTDIN_Rd.Close();
				m_ChildSTDOUT_Wr.Close();
				m_ChildSTDIN_Wr.Close();
				m_ChildSTDOUT_Rd.Close();
			}
		};

		#endif
	}
}

#endif	// __WBProcess_h__

//  End of Process.h
