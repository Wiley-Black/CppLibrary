/////////
//  Registry.h
//  Copyright (C) 2018 by Wiley Black
/////////

#ifndef __WBRegistry_h__
#define __WBRegistry_h__

#include "../wbFoundation.h"

#ifdef _WINDOWS

#include "Text/StringComparison.h"

namespace wb
{
	namespace Win32
	{		
		class Registry
		{
			static HKEY GetPredefinedBaseKey(string KeyPath, string& BaseKeyPart)
			{
				#define TEST_KEY(Text,PredefinedKey)		if (StartsWithNoCase(KeyPath, Text)) { BaseKeyPart = Text; return PredefinedKey; }
				TEST_KEY("HKEY_CLASSES_ROOT", HKEY_CLASSES_ROOT);
				TEST_KEY("HKCR", HKEY_CLASSES_ROOT);
				TEST_KEY("HKEY_CURRENT_USER", HKEY_CURRENT_USER);
				TEST_KEY("HKCU", HKEY_CURRENT_USER);
				TEST_KEY("HKEY_LOCAL_MACHINE", HKEY_LOCAL_MACHINE);
				TEST_KEY("HKLM", HKEY_LOCAL_MACHINE);
				TEST_KEY("HKEY_USERS", HKEY_USERS);
				TEST_KEY("HKU", HKEY_USERS);
				TEST_KEY("HKEY_CURRENT_CONFIG", HKEY_CURRENT_CONFIG);
				TEST_KEY("HKCC", HKEY_CURRENT_CONFIG);
				TEST_KEY("HKEY_CURRENT_USER_LOCAL_SETTINGS", HKEY_CURRENT_USER_LOCAL_SETTINGS);
				TEST_KEY("HKEY_PERFORMANCE_DATA", HKEY_PERFORMANCE_DATA);
				TEST_KEY("HKEY_PERFORMANCE_NLSTEXT", HKEY_PERFORMANCE_NLSTEXT);
				TEST_KEY("HKEY_PERFORMANCE_TEXT", HKEY_PERFORMANCE_TEXT);
				#undef TEST_KEY
				throw NotSupportedException("The requested registry key does not match any of the known, predefined base keys.");
			}

		public:
			
			static string GetValue(string keyName, string valueName, string defaultValue)
			{				
				// Retrieve one of the predefined key handles (the Win32/C++ interface uses these whereas .NET uses a single string for the entire subkey)...
				string BaseKeyText;
				HKEY BaseKey = GetPredefinedBaseKey(keyName, BaseKeyText);
				string SubKeyPath = keyName.substr(BaseKeyText.length());
				if (SubKeyPath.length() > 1 && (SubKeyPath[0] == '\\' || SubKeyPath[0] == '/')) SubKeyPath = SubKeyPath.substr(1);

				// Open the subkey...
				HKEY SubKey;
				LONG retcode;
				retcode = ::RegOpenKeyEx(BaseKey, to_osstring(SubKeyPath).c_str(), 0, KEY_QUERY_VALUE, &SubKey);
				if (retcode != ERROR_SUCCESS)
				{
					if (retcode == ERROR_FILE_NOT_FOUND) return defaultValue;
					Exception::ThrowFromWin32(retcode);					
				}
				try
				{
					// Query the type and size of the value and ensure that it's a string...
					DWORD dwType;
					DWORD DataLength = 0;
					retcode = ::RegQueryValueEx(SubKey, to_osstring(valueName).c_str(), NULL, &dwType, NULL, &DataLength);
					if (retcode != ERROR_SUCCESS && retcode != ERROR_MORE_DATA)
					{
						if (retcode == ERROR_FILE_NOT_FOUND) return defaultValue;
						Exception::ThrowFromWin32(retcode);
					}
					if (dwType != REG_SZ && dwType != REG_EXPAND_SZ)
						throw FormatException("Requested registry key did not contain a string type value.");

					// Retrieve the value data...
					// Note: DataLength is given in bytes, but the string will match UNICODE/not UNICODE, so it is an osstring.
					byte* pBuffer = new byte[DataLength];
					try
					{
						retcode = ::RegQueryValueEx(SubKey, to_osstring(valueName).c_str(), NULL, &dwType, pBuffer, &DataLength);
						if (retcode != ERROR_SUCCESS && retcode != ERROR_MORE_DATA)
						{
							if (retcode == ERROR_FILE_NOT_FOUND) return defaultValue;
							Exception::ThrowFromWin32(retcode);					
						}
						osstring ret_os = osstring((const TCHAR*)pBuffer, (DataLength / sizeof(TCHAR)));
						if (ret_os.length() > 0 && ret_os[ret_os.length()-1] == 0) ret_os = ret_os.substr(0, ret_os.length()-1);		// If the null-terminator became the final character, clobber it.
						return to_string(ret_os);
					}
					catch (...)
					{
						delete[] pBuffer;
						throw;
					}

					::RegCloseKey(SubKey);
				}
				catch(...)
				{
					::RegCloseKey(SubKey);
					throw;
				}				
			}

		};
		
	}
}

#endif	// _WINDOWS

#endif	// __WBRegistry_h__

//  End of Registry.h
