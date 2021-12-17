/////////
//  Environment.h
//  Copyright (C) 2014 by Wiley Black
/////////

#ifndef __WBEnvironment_h__
#define __WBEnvironment_h__

#include "../wbFoundation.h"
#include "../IO/Streams.h"
#include "../Foundation/STL/Collections/UnorderedMap.h"
#include "../Text/StringBuilder.h"

#ifndef _WINDOWS
#include <unistd.h>
#endif

namespace wb
{
	namespace sys
	{		

		class Environment
		{
			#ifdef _WINDOWS
			unordered_map<osstring, osstring> m_Environment;
			#endif

			osstring GetEnvironmentVariable(osstring key)
			{
				#ifdef _WINDOWS
				return m_Environment.at(key);
				#else
				char *pszValue = ::getenv(key.c_str());
				if (pszValue == nullptr) throw Exception("Environment variable '" + key + "' not found.");
				return pszValue;
				#endif
			}

		public:

			#ifdef _WINDOWS			// Can be defined for Linux, just isn't yet.
			Environment()
			{
				m_Environment = GetEnvironmentVariables();
			}

			static unordered_map<osstring, osstring> GetEnvironmentVariables()
			{
				LPTCH AllStrings = ::GetEnvironmentStrings();
				unordered_map<osstring, osstring> ret;

				TCHAR* psz = AllStrings;
				for (;;)
				{
					osstring	Key;
					osstring	Value;
					bool ParsingKey = true;
					while (*psz)
					{
						if (ParsingKey)
						{
							if (*psz == '=') { ParsingKey = false; }
							else Key += *psz;
						}
						else Value += *psz;						
						psz++;
					}
					ret[Key] = Value;
					psz++;
					if (!*psz) break;		// A double zero terminator delimits the end of environment block.
				}

				::FreeEnvironmentStrings(AllStrings);
				return ret;
			}
			
			#undef SetEnvironmentVariable
			static void SetEnvironmentVariable(osstring variable, osstring value)
			{
				#ifdef UNICODE
				if (!::SetEnvironmentVariableW(variable.c_str(), value.c_str())) Exception::ThrowFromWin32(::GetLastError());
				#else
				if (!::SetEnvironmentVariableA(variable.c_str(), value.c_str())) Exception::ThrowFromWin32(::GetLastError());
				#endif // !UNICODE				
			}
			#endif	// End _WINDOWS

			enum EnvSpecStyle
			{
				EnvSpec_Bash		= 0x0001,
				EnvSpec_Windows		= 0x0002,
				EnvSpec_All			= 0xFFFF
			};

			/// <summary>Replaces all environment variable substitutions within the string.  There are two differences from the .NET API version:
			///	1. This function is not static.  This is an optimization, in some platforms all environment variables must be retrieved prior to
			///    expanding the string.
			/// 2. More than just the %variable% format is accepted based on the SpecStyle parameter.
			/// </summary>
			string ExpandEnvironmentVariables(string text, EnvSpecStyle SpecStyle = EnvSpec_All)
			{				
				wb::text::StringBuilder sb;
				for (size_t pos = 0; pos < text.length(); )
				{
					if ((SpecStyle & EnvSpec_Bash) != 0 && text[pos] == '$')
					{
						if (pos+1 < text.length())
						{				
							if (text[pos+1] == '{')
							{
								size_t term = text.find('}', pos+2);
								if (term == string::npos) throw FormatException("Unterminated environment variable specifier.");
								// i.e. ${hello}
								//		01234567 has $ at pos=0, pos+2 = 2, term=7, term-(pos+2) = 5
								string env_var = text.substr(pos+2, term-(pos+2));								
								sb.Append(to_string(GetEnvironmentVariable(to_osstring(env_var))));
								pos = term + 1;
								continue;
							}							
							else
							{
								size_t term = pos+1;
								while (term < text.length())
								{									
									if ((text[term] < '0' || (text[term] > '9' && text[term] < 'A') || (text[term] > 'Z' && text[term] < 'a') || text[term] > 'z') && text[term] != '_') break;
									term++;
								}
								// i.e. $hello
								//		012345 has $ at pos=0, pos+1 = 1, term=6, term-(pos+1) = 5
								string env_var = text.substr(pos+1, term-(pos+1));
								sb.Append(to_string(GetEnvironmentVariable(to_osstring(env_var))));
								pos = term;
								continue;
							}
						}
						else throw FormatException("Invalid environment variable specifier.");
					}
					else if ((SpecStyle & EnvSpec_Windows) != 0 && text[pos] == '%')
					{						
						size_t term = text.find('%', pos+1);
						if (term == string::npos) throw FormatException("Unterminated environment variable specifier.");
						// i.e. %hello%
						//		0123456 has % at pos=0, pos+1 = 1, term=6, term-(pos+1) = 5
						string env_var = text.substr(pos+1, term-(pos+1));								
						sb.Append(to_string(GetEnvironmentVariable(to_osstring(env_var))));
						pos = term + 1;
						continue;
					}
					else 
					{
						sb.Append(text[pos]);
						pos++;
					}
				}
				return sb.ToString();
			}

			static int GetProcessorCount()
			{
				#ifdef _WINDOWS
				SYSTEM_INFO sysinfo;
				::GetSystemInfo(&sysinfo);
				return sysinfo.dwNumberOfProcessors;
				#else
				return sysconf(_SC_NPROCESSORS_ONLN);
				#endif
			}

		};
		
	}
}

#endif	// __WBEnvironment_h__

//  End of Environment.h
