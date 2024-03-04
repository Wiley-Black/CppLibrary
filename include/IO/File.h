/////////
//	File.h
//	Copyright (C) 2014 by Wiley Black
////

// Include wbFoundation.h ahead of the __WBFile_h__ exclusion in order to prevent dependency sequencing issues.
// wbFoundation.h must not do anything outside of its exclusion region, but File.h is not the first header to
// be included so it can use wbFoundation.h to enforce sequencing.
#include "../wbFoundation.h"

#ifndef __WBFile_h__
#define __WBFile_h__

#include <sys/stat.h>

#if !defined(_WINDOWS)
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#endif

#include "DateTime/DateTime.h"
#include "IO/MemoryStream.h"

namespace wb
{
	namespace io
	{
		class FileInfo;				// Forward declaration
		class DirectoryInfo;		// Forward declaration

		class File
		{
		public:

			static bool Exists(const osstring& strPath)
			{
				#ifdef _WINDOWS

				WIN32_FIND_DATA FindData;
				HANDLE hSearch = FindFirstFile(strPath.c_str(), &FindData);
				if (hSearch == INVALID_HANDLE_VALUE)
				{
					DWORD LastError = GetLastError();
					if (LastError == ERROR_FILE_NOT_FOUND || LastError == ERROR_PATH_NOT_FOUND) return false;
					Exception::ThrowFromWin32(LastError);
				}
				FindClose(hSearch);
				return true;

				#else

				struct stat buffer;
				int status = stat(strPath.c_str(), &buffer);
				if (status == 0) return true;
				// Verified that errno = ENOENT under Ubuntu when the file doesn't exist.  I didn't verify that
				// the directory being missing causes ENOTDIR, but it seems reasonable from the description.
				// Under both those cases, we return false because this is the normal "file not found" condition
				// we were looking for.
				if (errno != ENOENT && errno != ENOTDIR)
				{
					try { Exception::ThrowFromErrno(errno); }
					catch (std::exception& ex) { throw IOException("Unable to check existance of file '" + strPath + "':" + string(ex.what())); }
				}
				return false;

				#endif
			}

			static bool Exists(const string& sPath) { return Exists(to_osstring(sPath)); }

			static void Delete(const string& sPath)
			{
				// From the .NET API for File.Delete(): If the file to be deleted does not exist, no exception is thrown.

				#if defined(_WINDOWS)
				if (::DeleteFile(to_osstring(sPath).c_str()) == 0)
				{
					DWORD dwLastError = ::GetLastError();
					if (dwLastError == ERROR_FILE_NOT_FOUND) return;
					throw IOException();
				}
				#else				
				if (::unlink(sPath.c_str()) != 0) 
				{
					if (errno == ENOENT) return;
					try { Exception::ThrowFromErrno(errno); }
					catch (std::exception& ex) { throw IOException("Unable to delete file '" + sPath + "':" + string(ex.what())); }
				}
				#endif
			}

			static DateTime GetLastWriteTime(const string& sPath);

			#if defined(_WINDOWS)
			inline static void Copy(const string& SourcePath, const string& DestPath, bool Overwrite = false)
			{
				#ifdef _LINUX
				/** Untested, needs some work. **/
				int source = open(SourcePath.c_str(), O_RDONLY, 0);
				int dest = open(DestPath.c_str(), O_WRONLY | O_CREAT /*| O_TRUNC*/, 0644);

				// struct required, rationale: function stat() exists also
				struct stat stat_source;
				fstat(source, &stat_source);

				sendfile(dest, source, 0, stat_source.st_size);

				close(source);
				close(dest);
				#elif defined(_WINDOWS)
				if (!::CopyFile(to_osstring(SourcePath).c_str(), to_osstring(DestPath).c_str(), !Overwrite)) Exception::ThrowFromWin32(::GetLastError());
				#endif
			}
			#endif

			inline static void WriteAllText(string path, string contents);

			inline static string ReadAllText(string path);
		};

		class Directory
		{
		private:

			/// <summary>Creates all parts of the requested path that don't already exist, but does not retrieve the DirectoryInfo.</summary>
			static void CreateDirectoryNoEnum(const string& sPath);

		public:
			static bool Exists(const char* pszPath);
			static bool Exists(const string& sPath) { return Exists(sPath.c_str()); }			

			/// <summary>Creates all parts of the requested path.  If parts of the path already exist, they are used and not created but the DirectoryInfo is still retrieved.</summary>
			static DirectoryInfo CreateDirectory(const string& sPath);

			/// <summary>Changes to the requested directory.</summary>
			static void SetCurrentDirectory(const char* pszPath);			

			static void SetCurrentDirectory(string sPath) { SetCurrentDirectory(sPath.c_str()); }

			/// <summary>Retrieves the application's current working directory.</summary>
			static string GetCurrentDirectory();
		};
	}
}

//	Late Dependencies

#include "Path.h"
#include "FileInfo.h"
#include "FileStream.h"
#include "../Text/Encoding.h"
#include "../Text/StringBuilder.h"

namespace wb
{
	namespace io
	{
		/** Implementation - File **/

		inline /*static*/ DateTime File::GetLastWriteTime(const string& sPath)
		{
			string FullName = Path::StripTrailingSeparator(Path::ToAbsolutePath("", sPath));

			#ifdef _WINDOWS

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile(to_osstring(FullName).c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				if (GetLastError() == ERROR_FILE_NOT_FOUND) throw DirectoryNotFoundException();
				throw IOException();
			}				
			DateTime LastWriteTime = DateTime(FindData.ftLastWriteTime);
			FindClose(hSearch);				

			#else

			struct stat attrib;         // create a file attribute structure
			if (stat(FullName.c_str(), &attrib) != 0) throw DirectoryNotFoundException();							
			DateTime LastWriteTime = DateTime(attrib.st_mtime);

			#endif
			return LastWriteTime;
		}

		inline /*static*/ void File::WriteAllText(string path, string contents)
		{
			FileStream fs(path, FileMode::Create, FileAccess::ReadWrite);
			wb::text::StringBuilder sb(contents);
			fs.Write(sb.BaseAddress(), sb.GetLength());
		}

		inline /*static*/ string File::ReadAllText(string path)
		{
			FileInfo fi(path);
			FileStream fs(path, FileMode::Open, FileAccess::Read);
			wb::text::StringBuilder sb;
			if (sizeof(size_t) == sizeof(UInt64))			
				sb.SetLength((size_t)fi.GetLength());
			else
			{
				if (sizeof(size_t) == sizeof(UInt32))
				{
					if (fi.GetLength() > UInt32_MaxValue) throw NotSupportedException("32-bit implementation of ReadAllText cannot support a file larger than 4GB.");
					sb.SetLength((size_t)fi.GetLength());
				}
				else throw NotImplementedException("Unhandled definition of size_t format.");
			}
			fs.Read(sb.BaseAddress(), sb.GetLength());
			return sb.ToString();
		}

		/** Implementation - Directory **/

		inline /*static*/ bool Directory::Exists(const char* pszPath)
		{
			#ifdef _WINDOWS

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile(Path::StripTrailingSeparator(Path(pszPath)).to_osstring().c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				DWORD LastError = GetLastError();
				if (LastError == ERROR_FILE_NOT_FOUND || LastError == ERROR_PATH_NOT_FOUND) return false;
				Exception::ThrowFromWin32(LastError);
			}
			FindClose(hSearch);
			return (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) != 0;

			#else

			struct stat buffer;
			return (stat(pszPath, &buffer) == 0 && S_ISDIR(buffer.st_mode));

			#endif
		}

		inline /*static*/ string Directory::GetCurrentDirectory()
		{
			#ifdef _WINDOWS

			DWORD dwBufferLen = ::GetCurrentDirectory(0, nullptr);
			if (dwBufferLen != 0)
			{
				MemoryStream buf((dwBufferLen + 1) * sizeof(TCHAR));

				dwBufferLen = ::GetCurrentDirectory(dwBufferLen, (TCHAR*)buf.GetDirectAccess());
				if (dwBufferLen != 0)
				{
					osstring as_os((TCHAR*)buf.GetDirectAccess());
					string ret = wb::to_string(as_os);
					return ret;
				}
			}
			try { Exception::ThrowFromWin32(GetLastError()); }
			catch (std::exception& ex) { throw IOException("Unable to retrieve current working directory: " + string(ex.what())); }
			throw IOException("Unable to retrieve current working directory.");

			#else

			char* pwd = get_current_dir_name();
			if (pwd == nullptr)
			{
				try { Exception::ThrowFromErrno(errno); }
				catch (std::exception& ex) { throw IOException("Unable to retrieve current working directory: " + string(ex.what())); }
				throw IOException("Unable to retrieve current working directory.");
			}
			string ret(pwd);
			free(pwd);
			return ret;

			#endif
		}

		inline /*static*/ void Directory::CreateDirectoryNoEnum(const string& strPath)
		{
			// Recursively create any parents that don't already exist...			
			string Parent = Path::GetDirectory(strPath);
			if (Parent.length() >= 1)
				if (!Path::IsRoot(Parent) && !Directory::Exists(Parent)) CreateDirectoryNoEnum(Parent);

			// Now create this directory...
			#ifdef _WINDOWS
			if (::CreateDirectory(to_osstring(strPath).c_str(), nullptr) == 0)
			{
				DWORD dwError = GetLastError();
				if (dwError == ERROR_ALREADY_EXISTS) return;
				try { 										
					Exception::ThrowFromWin32(dwError); 
				}
				catch (std::exception& ex) { throw IOException("Unable to create directory '" + strPath + "':" + string(ex.what())); }
			}
			#else
			if (mkdir(Path::StripTrailingSeparator(strPath).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) != 0)
			{
				if (errno == EEXIST) return;
				try { Exception::ThrowFromErrno(errno); }
				catch (std::exception& ex) { throw IOException("Unable to create directory '" + strPath + "':" + string(ex.what())); }
			}
			#endif
		}

		inline /*static*/ DirectoryInfo Directory::CreateDirectory(const string& strPath)
		{
			CreateDirectoryNoEnum(strPath);
			return DirectoryInfo(strPath);
		}

		/// <summary>Changes to the requested directory.</summary>
		inline /*static*/ void Directory::SetCurrentDirectory(const char* pszPath)
		{				
			#ifdef _WINDOWS
			if (!::SetCurrentDirectory(to_osstring(pszPath).c_str()))
			{
				try { Exception::ThrowFromWin32(GetLastError()); }
				catch (std::exception& ex) { throw IOException("Unable to set current directory to path '" + string(pszPath) + "':" + string(ex.what())); }
			}
			#else
			int status = chdir(Path::StripTrailingSeparator(pszPath).c_str());
			if (status == 0) return;
			try { Exception::ThrowFromErrno(errno); }
			catch (std::exception& ex) { throw IOException("Unable to set current directory to path '" + string(pszPath) + "':" + string(ex.what())); }
			#endif
		}
	}
}

#endif	// __WBFile_h__

//	End of File.h

