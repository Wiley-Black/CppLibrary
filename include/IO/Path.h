/////////
//	Path.h
//	Copyright (C) 2014-2024 by Wiley Black
////

// Include wbFoundation.h ahead in order to prevent dependency sequencing issues.  wbFoundation.h must not do anything outside of its exclusion region.
#include "../wbFoundation.h"

#include <regex>

#ifndef __WBPath_h__
#define __WBPath_h__

namespace wb
{
	namespace io
	{
		class Path
		{
			void static replace_all(string& str, const char* pszFind, const char* pszReplace);

		public:

			#ifdef _WINDOWS
			enum { DirectorySeparatorChar = '\\' };
			#else
			enum { DirectorySeparatorChar = '/' };
			#endif

		private:							

			// String representation of the path.  May or may not contain trailing separators.
			wb::osstring m_str;

		public:

			Path() { }
			Path(const std::string& p) { m_str = wb::to_osstring(p); }
			Path(const std::wstring& p) { m_str = wb::to_osstring(p); }
			Path(const char* p) { m_str = wb::to_osstring(p); }
			Path(const wchar_t* p) { m_str = wb::to_osstring(p); }

			std::string to_string() const { return wb::to_string(m_str); }
			wb::osstring to_osstring() const { return m_str; }
			operator const std::string() const { return wb::to_string(m_str); }
			operator const std::wstring() const { return wb::to_wstring(m_str); }
			bool empty() const { return m_str.length() == 0; }
			bool operator!() const { return m_str.length() == 0; }

			// Path is a thin wrapper around wb::osstring, and can often be treated interchangably.
			size_t length() const { return m_str.length(); }
			wb::osstring substr(size_t pos = 0, size_t len = std::string::npos) const { return m_str.substr(pos, len); }
			inline oschar& operator[] (size_t pos) { return m_str[pos]; }
			inline const oschar& operator[] (size_t pos) const { return m_str[pos]; }
			inline size_t find(const osstring& str, size_t pos = 0) const { return m_str.find(str, pos); }
			inline size_t find(oschar ch, size_t pos = 0) const { return m_str.find(ch, pos); }
			inline size_t rfind(const osstring& str, size_t pos = 0) const { return m_str.rfind(str, pos); }
			inline size_t rfind(oschar ch, size_t pos = 0) const { return m_str.rfind(ch, pos); }

			Path operator/(const Path& other) const { return Path::Join(m_str, other.m_str); }
			Path operator+(const string& other) const {
				Path ret(*this);
				ret.m_str += wb::to_osstring(other);
				return ret;
			}

			inline osstring GetFullName() const { return to_osstring(); }

			inline osstring GetFileName() const
			{
				if (length() < 1) return os("");
				if (length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)length() - 1; ii >= 0; ii--)
				{
					if (m_str[ii] == (oschar)'/' || m_str[ii] == (oschar)'\\') return m_str.substr(ii + 1);
				}
				return m_str;
			}

			inline static osstring GetFileName(Path path) { return path.GetFileName(); }

			/// <summary>
			/// Returns the file extension, defined as the last occurrence of '.' in the string.
			/// The returned file extension will include the dot, i.e. ".bmp", as long as there
			/// was a file extension.  A path without a dot as the last piece of the path will
			/// return "".
			/// </summary>
			inline osstring GetExtension() const
			{
				if (length() < 1) return os("");
				if (length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)length() - 1; ii >= 0; ii--)
				{
					if (m_str[ii] == (oschar)'.') return m_str.substr(ii);
					if (m_str[ii] == (oschar)'/' || m_str[ii] == (oschar)'\\') return os("");
				}
				return os("");
			}

			/// <summary>
			/// Returns the file extension, defined as the last occurrence of '.' in the string.
			/// The returned file extension will include the dot, i.e. ".bmp", as long as there
			/// was a file extension.  A path without a dot as the last piece of the path will
			/// return "".
			/// </summary>
			inline static osstring GetExtension(Path path) { return path.GetExtension(); }

			/// <summary>
			/// Returns the file extension, defined as the last occurrence of '.' in the string.
			/// The returned file extension will include the dot, i.e. ".bmp", as long as there
			/// was a file extension.  A path without a dot as the last piece of the path will
			/// return "".
			/// </summary>
			inline static string GetExtension(string path)
			{
				if (path.length() < 1) return "";
				if (path.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)path.length() - 1; ii >= 0; ii--)
				{
					if (path[ii] == '.') return path.substr(ii);
					if (path[ii] == '/' || path[ii] == '\\') return "";
				}
				return "";
			}
			
			inline osstring GetFileNameWithoutExtension() const
			{
				Path path = GetFileName();
				if (path.length() < 1) return os("");
				if (path.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)path.length() - 1; ii >= 0; ii--)
				{
					if (path[ii] == (oschar)'.') return path.substr(0, ii);
				}
				return path;
			}			

			inline static osstring GetFileNameWithoutExtension(Path path) { return path.GetFileNameWithoutExtension(); }

			inline Path WithExtension(const string& new_extension) const
			{
				string new_ext = new_extension;
				if (new_ext.length() > 0 && new_ext[0] != '.') new_ext = string(".") + new_extension;
				return GetDirectory() / (GetFileNameWithoutExtension() + wb::to_osstring(new_ext));
			}

			inline Path GetDirectory() const
			{
				if (m_str.length() < 1) return "";
				if (m_str.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)m_str.length() - 2; ii >= 0; ii--)		// Subtle: Start at length-2, so that a trailing character is ignored.
				{
					if (m_str[ii] == '/' || m_str[ii] == '\\') return m_str.substr(0, ii);
				}
				return "";
			}

			inline static string GetDirectory(string path)
			{
				if (path.length() < 1) return "";
				if (path.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)path.length() - 2; ii >= 0; ii--)		// Subtle: Start at length-2, so that a trailing character is ignored.
				{					
					if (path[ii] == '/' || path[ii] == '\\') return path.substr(0, ii);
				}
				return "";
			}

			inline static wstring GetDirectory(wstring path)
			{
				if (path.length() < 1) return L"";
				if (path.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				for (int ii = (int)path.length() - 2; ii >= 0; ii--)		// Subtle: Start at length-2, so that a trailing character is ignored.
				{
					if (path[ii] == L'/' || path[ii] == L'\\') return path.substr(0, ii);
				}
				return L"";
			}

			inline static Path GetDirectory(Path path) { return path.GetDirectory(); }

			inline bool IsRoot()
			{
				if (m_str.length() == 1 && m_str[0] == DirectorySeparatorChar) return true;
				#ifdef _WINDOWS
				if ((m_str.length() == 2 && m_str[1] == ':')
					|| (m_str.length() == 3 && m_str[1] == ':' && m_str[2] == DirectorySeparatorChar)) return true;
				#endif
				return false;
			}

			inline static bool IsRoot(string path)
			{
				if (path.length() == 1 && path[0] == DirectorySeparatorChar) return true;
				#ifdef _WINDOWS
				if ((path.length() == 2 && path[1] == ':')
				 || (path.length() == 3 && path[1] == ':' && path[2] == DirectorySeparatorChar)) return true;
				#endif
				return false;
			}

			inline static bool IsRoot(wstring path)
			{
				if (path.length() == 1 && path[0] == (wchar_t)DirectorySeparatorChar) return true;
#ifdef _WINDOWS
				if ((path.length() == 2 && path[1] == L':')
					|| (path.length() == 3 && path[1] == L':' && path[2] == (wchar_t)DirectorySeparatorChar)) return true;
#endif
				return false;
			}

			inline static bool IsRoot(Path path) { return path.IsRoot(); }

			bool IsAbsolutePath();
			static bool IsAbsolutePath(Path path) { return path.IsAbsolutePath(); }

			/// <summary>Converts a path to a base path through the following actions:
			///		1. If the path is relative, then prefix the base to it.  If the path is already absolute, then 'base' is ignored.
			///     2. If the path is still relative, then attach it to the current working directory.
			///		3. Replace any occurrances of . or .. with the proper folder.
			///		4. Replace any double separators with a single one.
			/// Steps 2 and 3 can be run on an already-rooted path in order to simplify the path expression and ensure consistency.  To
			/// accomplish this, call ToAbsolutePath() with an empty base string.</summary>
			static Path ToAbsolutePath(Path base, Path path);

			/// <summary>Retrieves a path given relative to a base path.  For example, if the base is "C:\Example" and the path
			/// is "C:\Example\SubDir\MyFile.txt" then the returned Path is "SubDir\MyFile.txt".  If path is not an absolute
			/// path or is not contained within the base path then an exception is thrown.  Both base and path will be normalized
			/// by ToRelativePath() to a consistent format for comparison before any other processing.</summary>
			static Path ToRelativePath(Path base, Path path);		
						
			static Path Join(Path path1, Path path2);
			static Path Join(Path path1, Path path2, Path path3);
			static Path Join(Path path1, Path path2, Path path3, Path path4);
			static Path Join(Path path1, Path path2, Path path3, Path path4, Path path5);
			static Path Join(Path path1, Path path2, Path path3, Path path4, Path path5, Path path6);

			/// <summary>Retrieves the user's temporary directory.  This implementation also ensures that the directory exists.  If it does not exist and cannot be created, an exception is
			/// thrown.</summary>
			static Path GetTempPath();

			#ifdef _WINDOWS		// Needs a linux implementation.
			static Path GetTempFileName();
			#endif			
		
			bool Exists();

			/// <summary>
			/// IsDirectory() returns true if the path references an existing directory.  If the path does not exist or the
			/// path references a file and not a directory, then false is returned.
			/// </summary>			
			bool IsDirectory();

			inline static bool HasTrailingSeparator(string path)
			{				
				return (path.length() > 0 && path[path.length() - 1] == DirectorySeparatorChar);
			}

			inline static bool HasTrailingSeparator(wstring path)
			{
				return (path.length() > 0 && path[path.length() - 1] == (wchar_t)DirectorySeparatorChar);
			}

			inline static bool HasTrailingSeparator(Path path) { return HasTrailingSeparator(path.m_str); }

			inline static bool HasLeadingSeparator(string path)
			{				
				return (path.length() > 0 && path[0] == DirectorySeparatorChar);
			}

			inline static bool HasLeadingSeparator(wstring path)
			{
				return (path.length() > 0 && path[0] == (wchar_t)DirectorySeparatorChar);
			}

			inline static bool HasLeadingSeparator(Path path) { return HasLeadingSeparator(path.m_str); }

			inline Path StripTrailingSeparator()
			{
				osstring sPath = m_str;
				while (sPath.length() > 0 && sPath[sPath.length() - 1] == (oschar)DirectorySeparatorChar) sPath = sPath.substr(0, sPath.length() - 1);
				return sPath;
			}

			inline static Path StripTrailingSeparator(Path path) { return path.StripTrailingSeparator(); }			

			inline static string StripLeadingSeparator(string path)
			{				
				while (path.length() > 0 && path[0] == DirectorySeparatorChar) path = path.substr(1);
				return path;
			}

			inline static wstring StripLeadingSeparator(wstring path)
			{
				while (path.length() > 0 && path[0] == (wchar_t)DirectorySeparatorChar) path = path.substr(1);
				return path;
			}

			inline static Path StripLeadingSeparator(Path path) { return Path(StripLeadingSeparator(path.m_str)); }

			inline static string EnsureTrailingSeparator(string path)
			{
				if (path.length() > 0 && path[path.length() - 1] != DirectorySeparatorChar) 
					return path + (char)DirectorySeparatorChar;
				else
					return path;
			}									

			inline static wstring EnsureTrailingSeparator(wstring path)
			{
				if (path.length() > 0 && path[path.length() - 1] != (wchar_t)DirectorySeparatorChar)
					return path + (wchar_t)DirectorySeparatorChar;
				else
					return path;
			}

			inline static Path EnsureTrailingSeparator(Path path) { return EnsureTrailingSeparator(path.m_str); }

			/// <summary>Checks whether a given path or filename patches a wildcard pattern.  The characters ? and * are supported.</summary>
			inline static bool IsWildcardMatch(string pattern, string path, bool CaseSensitive /*= true*/);		

			inline void CreateDirectory(bool CreateParents = false, bool ExistsOk = true);

			inline void Delete(bool ok_if_missing = false, bool recursive = false);
		};
	}
}

// Resolve declarations
#include "File.h"
#include "FileInfo.h"

namespace wb { namespace io {
		using namespace std;

		/** Implementation - Path **/		

		inline bool Path::Exists()
		{
			return wb::io::File::Exists(m_str);
		}

		inline bool Path::IsDirectory()
		{
			#ifdef _WINDOWS
			DWORD FileAttributes = GetFileAttributes(m_str.c_str());
			if (FileAttributes == INVALID_FILE_ATTRIBUTES)
			{
				DWORD ErrCode = GetLastError();
				// ERROR_BAD_NETPATH is returned if the path is a network share.  One could debate whether
				// a network share is considered a directory, but since GetFileAttributes() doesn't, I won't
				// here either.
				if (ErrCode == ERROR_BAD_NETPATH) return false;
				if (ErrCode == ERROR_FILE_NOT_FOUND || ErrCode == ERROR_PATH_NOT_FOUND) return false;
				Exception::ThrowFromWin32(ErrCode);
			}
			return (FileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
			#else
			struct stat sb;
			return (stat(m_str.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode));
			#endif
		}

		inline bool Path::IsAbsolutePath()
		{
			if (m_str.length() < 1) return false;

			#ifdef _WINDOWS				
			if (m_str.length() < 2) return false;
			if (m_str[1] == ':') return true;
			if (m_str.length() < 3) return false;
			if (m_str[0] == '\\' && m_str[1] == '\\') return true;
			//if (path[0] == '/' && path[1] == '/') return true;			// Some mixed thoughts on this one- if there are accidently double directory separators this might be relative.  Or not.
			return false;

			#else
			if (m_str[0] == '/' || m_str[0] == '\\') return true;
			return false;
			#endif
		}

		/// <summary>This implementation also ensures that the directory exists.  If it does not exist and cannot be created, an exception is
		/// thrown.</summary>
		inline /*static*/ Path Path::GetTempPath()
		{
			try
			{
				#ifdef _WINDOWS
				for (int retry = 0; retry < 10; retry++)	// Just in case the path changes while we're trying to retrieve it and gets longer.
				{
					DWORD nLength = ::GetTempPath(0, nullptr);
					if (nLength == 0) Exception::ThrowFromWin32(::GetLastError());
					TCHAR* pszBuffer = new TCHAR[nLength+1];
					DWORD nResult = ::GetTempPath(nLength+1, pszBuffer);
					if (nResult == 0) { delete[] pszBuffer; Exception::ThrowFromWin32(::GetLastError()); }
					if (nResult > nLength+1) { delete[] pszBuffer; continue; }
					auto ret = wb::to_osstring(pszBuffer);
					delete[] pszBuffer;
					if (Directory::Exists(ret)) return ret;
					Directory::CreateDirectory(ret);
					return Path(ret);
				}
				throw Exception("Unable to retrieve temporary path after 10 tries.");
				#else
				string ret;

				// According to glibc documentation, we try to retrieve the temporary path in the following sequence:					
				// 1. The environment variable TMPDIR, if it is defined. For security reasons this only happens if the program is not SUID or SGID enabled.
				char* pszTmpDir = getenv("TMPDIR");
				if (pszTmpDir != nullptr && strlen(pszTmpDir) > 0) ret = pszTmpDir;					
				// 2. The value of the P_tmpdir macro.
				#ifdef P_tmpdir
				else if (P_tmpdir != nullptr && Directory::Exists(P_tmpdir)) ret = P_tmpdir;
				#endif
				// 3. The directory /tmp.
				else if (Directory::Exists("/tmp")) ret = "/tmp";
				#ifdef P_tmpdir
				else ret = P_tmpdir;
				#else
				else ret = "/tmp";
				#endif

				if (Directory::Exists(ret)) return ret;
				Directory::CreateDirectory(ret);
				return Path(ret);
				#endif
			}
			catch (std::exception& ex)
			{
				throw IOException("Unable to retrieve temporary path: " + string(ex.what()));
			}				
		}

		#ifdef _WINDOWS
		inline /*static*/ Path Path::GetTempFileName()
		{
			try
			{
				TCHAR TempFileName[MAX_PATH+1];				
				osstring TempPath = GetTempPath().to_osstring();
				if (!::GetTempFileName(TempPath.c_str(), wb::to_osstring("wb_temp_").c_str(), 0, TempFileName))
				{
					Exception::ThrowFromWin32(::GetLastError());
				}
				return Path(wb::to_string(TempFileName));
			}
			catch (std::exception& ex)
			{
				throw IOException("Unable to retrieve temporary path: " + string(ex.what()));
			}				
		}
		#endif

		inline /*static*/ Path Path::ToAbsolutePath(Path base, Path path)
		{
			#ifdef _WINDOWS
			static const oschar* PreviousFolderMatch = os("\\..");
			static const oschar* CurrentFolderMatch = os("\\.");
			static const oschar* DoubleSepMatch = os("\\\\");
			#else
			static const oschar* PreviousFolderMatch = os("/..");
			static const oschar* CurrentFolderMatch = os("/.");
			static const oschar* DoubleSepMatch = os("//");
			#endif

			if (path.length() < 1) return base;
			//if (path[0] != (char)DirectorySeparatorChar) path = Path::Join(base, path);
			if (!Path::IsAbsolutePath(path)) path = Path::Join(base, path);

			// If the path is still a relative path, then assume it is relative to the current
			// working directory and make it absolute as such.
			if (!Path::IsAbsolutePath(path)) path = Path::Join(wb::io::Directory::GetCurrentDirectory(), path);			

			// Normalize all directory separators to the current operating system's preference
			// TODO: Will this cause any trouble with UNC paths under Linux?  Can you use UNC start symbols in Linux?
			for (;;)
			{
				#ifdef _WINDOWS
				size_t index = path.find((oschar)'/');
				#else
				size_t index = path.find((oschar)'\\');
				#endif
				if (index == string::npos) break;
				path[index] = DirectorySeparatorChar;
			}

			//string filename = Path::GetFileName(path);
			//path = Path::GetDirectory(path);				// Operate without the filename for now.				

			// Process any .. occurrances.
			for (;;)
			{
				size_t index = path.find(PreviousFolderMatch);
				if (index == string::npos) break;
				if (index == 0) throw FormatException("Path to parent directory of root requested in path resolution.");

				osstring before = path.substr(0, index);				
				osstring after = path.substr(index+3);
				path = Path::Join(Path::GetDirectory(before), after);
			}
				
			// Process any . occurrances.
			for (;;)
			{
				size_t index = path.find(CurrentFolderMatch);
				if (index == string::npos) break;

				osstring before = path.substr(0, index);
				osstring after = path.substr(index+2);
				path = Path::Join(before, after);
			}				

			// Remove any double separator occurrances, except UNC paths.
			for (;;)
			{
				size_t index = path.rfind(DoubleSepMatch);
				if (index == string::npos || index == 0) break;		// In combination with using rfind(), this protects UNC paths.

				path = path.substr(0, index) + path.substr(index+1);
			}

			return path;
		}

		inline /*static*/ Path Path::ToRelativePath(Path base, Path path)
		{
			// Simplify both base and path so that they will have a consistent format
			base = ToAbsolutePath("", base);
			path = ToAbsolutePath("", path);
			base = StripTrailingSeparator(base);
			path = StripTrailingSeparator(path);
		
			if (!wb::StartsWithNoCase(path, base)) throw FormatException("Cannot specify path '" + path.to_string() + "' relative to base path '" + base.to_string() + "'.");
			return StripLeadingSeparator(path.substr(base.length()));
		}

		inline /*static*/ void Path::replace_all(string& str, const char* pszFind, const char* pszReplace)
		{			
			string strFind = pszFind;
			string strReplace = pszReplace;
			
			for (size_t pos = 0; pos < str.length();)
			{
				size_t iFound = str.find(strFind, pos);
				if (iFound == string::npos) break;
				str.replace(iFound, strFind.length(), strReplace);
				pos = iFound + strReplace.length();
			}
		}

		inline /*static*/ bool Path::IsWildcardMatch(string pattern, string path, bool CaseSensitive /*= true*/)
		{					
			// Escape all regex special chars
			replace_all(pattern, "\\", "\\\\");
			replace_all(pattern, "^", "\\^");
			replace_all(pattern, ".", "\\.");
			replace_all(pattern, "$", "\\$");
			replace_all(pattern, "|", "\\|");
			replace_all(pattern, "(", "\\(");
			replace_all(pattern, ")", "\\)");
			replace_all(pattern, "[", "\\[");
			replace_all(pattern, "]", "\\]");
			replace_all(pattern, "*", "\\*");
			replace_all(pattern, "+", "\\+");
			replace_all(pattern, "?", "\\?");
			replace_all(pattern, "/", "\\/");

			// Convert chars '*?' back to their regex equivalents
			replace_all(pattern, "\\?", ".");
			replace_all(pattern, "\\*", ".*");

			if (!CaseSensitive)
			{
				std::regex regex_pattern(pattern);
				return std::regex_match(path, regex_pattern);
			}
			else
			{
				std::regex regex_pattern(pattern, std::regex::icase);
				return std::regex_match(path, regex_pattern);
			}			
		}

		inline /*static*/ Path Path::Join(Path path1, Path path2)
		{
			if (path1.length() < 1) return path2;
			if (path2.length() < 1) return path1;				

			// Remove "." in path2 if found...
			for (;;)
			{
				size_t ii = path2.find(os("/./"));
				if (ii != string::npos) {
					if (ii > 0) path2 = path2.substr(0, ii) + path2.substr(ii+3);
					else path2 = path2.substr(3);
					continue;
				}
				ii = path2.find(os("./"));
				if (ii == 0) { path2 = path2.substr(2); continue; }
				if (path2.length() == 1 && path2[0] == (oschar)'.') return path1;
				if (path2.length() < 1) return path1;
				break;
			}

			if (path1[path1.length() - 1] == (oschar)DirectorySeparatorChar)
			{
				if (path2[0] == (oschar)DirectorySeparatorChar)					
					return path1.m_str + path2.substr(1);
				else
					return path1.m_str + path2.m_str;
			}
			else
			{
				if (path2[0] == (oschar)DirectorySeparatorChar)
					return path1.m_str + path2.m_str;
				else
					return path1.m_str + (oschar)DirectorySeparatorChar + path2.m_str;
			}
		}

		inline /*static*/ Path Path::Join(Path path1, Path path2, Path path3) { return Join(Join(path1,path2),path3); }
		inline /*static*/ Path Path::Join(Path path1, Path path2, Path path3, Path path4) { return Join(Join(path1,path2,path3),path4); }
		inline /*static*/ Path Path::Join(Path path1, Path path2, Path path3, Path path4, Path path5) { return Join(Join(path1,path2,path3,path4),path5); }
		inline /*static*/ Path Path::Join(Path path1, Path path2, Path path3, Path path4, Path path5, Path path6) { return Join(Join(path1,path2,path3,path4,path5),path6); }		

		inline void Path::CreateDirectory(bool CreateParents /*= false*/, bool ExistsOk /*= true*/)
		{
			if (ExistsOk)
			{
				if (Directory::Exists(*this)) return;
			}

			if (CreateParents)
			{
				// Recursively create any parents that don't already exist...			
				auto Parent = GetDirectory();
				if (Parent.length() >= 1)
					if (!Path::IsRoot(Parent) && !Directory::Exists(Parent)) Path(Parent).CreateDirectory(CreateParents, true);
			}

			auto Parent = GetDirectory();
			if (!Directory::Exists(Parent))			
				throw IOException("Unable to create directory '" + to_string() + "': parent directory does not exist.");			

			// Now create this directory...
			#ifdef _WINDOWS
			if (::CreateDirectory(m_str.c_str(), nullptr) == 0)
			{
				DWORD dwError = GetLastError();
				if (dwError == ERROR_ALREADY_EXISTS) return;
				try {
					Exception::ThrowFromWin32(dwError);
				}
				catch (std::exception& ex) { throw IOException("Unable to create directory '" + to_string() + "':" + string(ex.what())); }
			}
			#else
			if (mkdir(Path::StripTrailingSeparator(m_str).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
			{
				if (errno == EEXIST) return;
				try { Exception::ThrowFromErrno(errno); }
				catch (std::exception& ex) { throw IOException("Unable to create directory '" + strPath + "':" + string(ex.what())); }
			}
			#endif
		}

		inline void Path::Delete(bool ok_if_missing /*= false*/, bool recursive /*= false*/)
		{
			if (!IsDirectory())
			{
				if (!ok_if_missing)
				{
					if (!File::Exists(m_str))
						throw FileNotFoundException();
				}
				File::Delete(m_str);
			}
			else
			{
				if (!ok_if_missing)
				{
					if (!Directory::Exists(m_str))
						throw DirectoryNotFoundException();
				}
				Directory::Delete(m_str, recursive);
			}
		}

		inline std::string to_string(const Path& pth)
		{
			return pth.to_string();
		}

		inline std::ostream& operator<<(std::ostream& os, const Path& pth)
		{
			os << pth.to_string();
			return os;
		}
} }

#endif	// __WBPath_h__

//	End of Path.h

