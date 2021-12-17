/////////
//	Diagnostics.h
//	Copyright (C) 2017 by Wiley Black
////
//	Provides miscellaneous diagnostic functions.
////

#ifndef __WBDiagnostics_h__
#define __WBDiagnostics_h__

/** Dependencies **/

#include "../wbFoundation.h"

#ifndef _WINDOWS
#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
#include <signal.h>

#include "Text/StringBuilder.h"
#endif

namespace wb { namespace diagnostics {

	/// <summary>
	/// ValidObject is a troubleshooting tool where a 4-byte signature code is attached to every object, initialized
	/// at construction.  The ValidateObject() call can be made on the object at any time to ensure that a valid,
	/// properly constructed object is in use.  A string is also provided that can be used to label an object.
	/// </summary>
	/// <seealso>DebugValidObject</seealso>
	class ValidObject
	{		
		UInt32 m_ValidCode;
		enum { TargetCode = 0x12345678 };
		enum { DestroyedCode = 0x0011DEAD };
		enum { MovedCode = 0x0022DEAD };
		enum { InvalidMoveCode = 0x0033DEAD };

		std::string		m_ObjectName;

	public:

		ValidObject()			
			: m_ValidCode(TargetCode)			
		{
		}

		ValidObject(const ValidObject& cp)
		{
			m_ValidCode = cp.m_ValidCode;
			m_ObjectName = "CCopy of " + cp.m_ObjectName;
		}

		ValidObject& operator=(const ValidObject& cp)
		{
			m_ValidCode = cp.m_ValidCode;
			m_ObjectName = "Copy of " + cp.m_ObjectName;
			return *this;
		}

		ValidObject(ValidObject&& mv) noexcept
		{
			m_ValidCode = (mv.m_ValidCode == TargetCode) ? TargetCode : InvalidMoveCode;
			mv.m_ValidCode = MovedCode;
			m_ObjectName = "CMoved from " + mv.m_ObjectName;
		}

		ValidObject& operator=(ValidObject&& mv) noexcept
		{
			m_ValidCode = (mv.m_ValidCode == TargetCode) ? TargetCode : InvalidMoveCode;
			mv.m_ValidCode = MovedCode;
			m_ObjectName = "Moved from " + mv.m_ObjectName;
			return *this;
		}

		~ValidObject() noexcept(false)
		{
			if (m_ValidCode == MovedCode) {
				m_ValidCode = DestroyedCode;
				return;
			}
			if (m_ValidCode == InvalidMoveCode) throw Exception("Destructor called on object that was transferred from an invalid source in a move operation.");
			if (m_ValidCode == DestroyedCode) throw Exception("Destructor called twice on object.");			
			if (m_ValidCode != TargetCode) throw Exception("Destructor called on invalid object.");
			m_ValidCode = DestroyedCode;
		}
		
		void ValidateObject() {
			if (m_ValidCode != TargetCode) {
				if (m_ValidCode == DestroyedCode) throw Exception("Attempt to ValidateObject() after destruction.");
				if (m_ValidCode == MovedCode) throw Exception("Attempt to ValidateObject() after an object was moved into another object.");
				if (m_ValidCode == InvalidMoveCode) throw Exception("Attempt to ValidObject() that was transferred from an invalid source in a move operation.");
				throw Exception("Invalid object, failed validation code.");
			}
		}

		void SetObjectName(const string& strLabel) 
		{ 
			m_ObjectName = strLabel;
		}

		string GetObjectName() const { return m_ObjectName; }
	};
	
	/// <summary>
	/// DebugValidObject is a variation on ValidObject that is only applied in _DEBUG builds.  For any other
	/// build, the 4-byte signature is absent and the ValidObject() call is available but does nothing.
	/// </summary>
	class DebugValidObject
		#ifdef _DEBUG
		: public ValidObject
		#endif
	{
	public:
		#ifndef _DEBUG
		void ValidateObject() { }
		void SetObjectName(const string& strLabel) { }
		string GetObjectName() const { return ""; }
		#endif
	};

#ifndef _WINDOWS

	// Taken from:
		// stacktrace.h (c) 2008, Timo Bingmann from http://idlebox.net/
		// published under the WTFPL v2.0
#if 0
	inline string GetCallStack(unsigned int max_frames = 63)
	{
		wb::text::StringBuilder sb;
		sb.AppendLine("Call Stack:");

		// storage array for stack trace address data
		void* addrlist[max_frames+1];

		// retrieve current stack addresses
		int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

		if (addrlen == 0) {
			sb.AppendLine("  <empty, possibly corrupt>");
			return sb.ToString();
		}

		// resolve addresses into strings containing "filename(function+address)",
		// this array must be free()-ed
		char** symbollist = backtrace_symbols(addrlist, addrlen);
		try
		{
			// allocate string which will be filled with the demangled function name
			size_t funcnamesize = 2048;
			char* funcname = (char*)malloc(funcnamesize);

			try
			{
				// iterate over the returned symbol lines. skip the first, it is the
				// address of this function.
				for (int i = 1; i < addrlen; i++)
				{
					char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

					// find parentheses and +address offset surrounding the mangled name:
					// ./module(function+0x15c) [0x8048a6d]
					for (char *p = symbollist[i]; *p; ++p)
					{
						if (*p == '(')
							begin_name = p;
						else if (*p == '+')
							begin_offset = p;
						else if (*p == ')' && begin_offset) {
							end_offset = p;
							break;
						}
					}

					if (begin_name && begin_offset && end_offset
						&& begin_name < begin_offset)
					{
						*begin_name++ = '\0';
						*begin_offset++ = '\0';
						*end_offset = '\0';

						// mangled name is now in [begin_name, begin_offset) and caller
						// offset in [begin_offset, end_offset). now apply
						// __cxa_demangle():

						int status;
						char* ret = abi::__cxa_demangle(begin_name,
										funcname, &funcnamesize, &status);
						if (status == 0) {
							funcname = ret; // use possibly realloc()-ed string
							sb.AppendLine("  " + string(symbollist[i]) + " : " + string(funcname) + "+" + begin_offset);
						}
						else {
							// demangling failed. Output function name as a C function with
							// no arguments.
							sb.AppendLine("  " + string(symbollist[i]) + " : " + string(begin_name) + "()+" + begin_offset);						
						}
					}
					else
					{
						// couldn't parse the line? print the whole line.
						sb.AppendLine("  " + string(symbollist[i]));
					}
				}
			}
			catch (...)
			{
				free(funcname);
				throw;
			}

			free(funcname);
		}
		catch (...)
		{
			free(symbollist);
			throw;
		}		
		free(symbollist);

		return sb.ToString();
	}
#endif
		
	inline void PrintCallStack(char* pszExecutable, unsigned int max_frames = 63)
	{		
		// storage array for stack trace address data
		void* addrlist[max_frames+1];

		// retrieve current stack addresses
		int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

		if (addrlen == 0) printf("  <empty, possibly corrupt backtrace>\n");

		for (int ii = 0; ii < addrlen; ii++)
		{
			char syscom[4096];
			sprintf(syscom,"addr2line %p -e %s", addrlist[ii], pszExecutable); //last parameter is the name of this app
			int val = system(syscom);
            if (val < 0) printf("  Unable to call system command addr2line.  Unable to display call stack.\n");
		}
	}		

#endif

} }// End namespace

#endif // __WBDiagnostics_h__

//	End of Diagnostics.h

