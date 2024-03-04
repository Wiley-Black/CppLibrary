/////////
//	FileInfo.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBFileInfo_h__
#define __WBFileInfo_h__

/** Table of contents **/

namespace wb {
	namespace io {
		class FileInfo;
		class DirectoryInfo;
	}
}

/** Dependencies **/

#include "../wbFoundation.h"
#include "../Text/StringComparison.h"
#include "../DateTime/DateTime.h"

#if !defined(_WINDOWS)
#include <sys/stat.h>
#include <dirent.h>
#endif

/** Content **/

namespace wb
{
	namespace io
	{
		class FileInfo
		{
			friend class DirectoryInfo;
		protected:			

			/// <summary>Stores the full path to the file or directory.  In the case of a directory,
			/// FullName should not be stored with the trailing directory separator.</summary>
			Path FullName;
			DateTime CreationTime;
			DateTime LastWriteTime;
			UInt64 Length;
		public:			
			FileInfo() : Length(0) { }
			FileInfo(const string& Path);			

			bool IsEmpty() const { return FullName.empty(); }
			bool operator!() const { return FullName.empty(); }

			/// <summary>
			/// Retrieves the full path of the directory or file.  For example, "C:\Path\Examples.xml".
			/// </summary>
			Path GetFullName() const { return FullName; }

			/// <summary>
			/// Retrieves the filename of the directory or file.  For example, "Examples.xml".
			/// </summary>
			osstring GetName() const;

			/// <summary>
			/// Retrieves the extension of the directory or file.  For example, ".xml".  The dot will be
			/// included as long as the file has a dot in the name.
			/// </summary>			
			osstring GetExtension() const;
			DateTime GetCreationTime() const { return CreationTime; }
			DateTime GetLastWriteTime() const { return LastWriteTime; }
			UInt64 GetLength() const { return Length; }
			bool Exists() const;
		};

		class DirectoryInfo : public FileInfo
		{
		protected:
			DirectoryInfo() { }

		public:			
			DirectoryInfo(const string& Path);

			vector<FileInfo>		EnumerateFiles() const;
			vector<FileInfo>		EnumerateFiles(string searchPattern) const;
			vector<DirectoryInfo>	EnumerateDirectories() const;
		};				
	}
}

#include "Path.h"

namespace wb
{
	namespace io
	{
		/** Implementation - FileInfo **/

		inline FileInfo::FileInfo(const string& sPath)
		{
			FullName = Path::StripTrailingSeparator(Path::ToAbsolutePath("", sPath));

			#ifdef _WINDOWS

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile(FullName.to_osstring().c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				if (GetLastError() == ERROR_FILE_NOT_FOUND) throw DirectoryNotFoundException();
				throw IOException();
			}
			CreationTime = DateTime(FindData.ftCreationTime);
			LastWriteTime = DateTime(FindData.ftLastWriteTime);
			Length = MakeU64(FindData.nFileSizeLow, FindData.nFileSizeHigh);
			FindClose(hSearch);

			#else

			struct stat attrib;         // create a file attribute structure
			if (stat(FullName.c_str(), &attrib) != 0) throw DirectoryNotFoundException();			
			CreationTime = DateTime(attrib.st_ctime);
			LastWriteTime = DateTime(attrib.st_mtime);
			Length = attrib.st_size;

			#endif
		}

		inline osstring FileInfo::GetName() const { return Path::GetFileName(FullName); }
		inline osstring FileInfo::GetExtension() const { return Path::GetExtension(FullName); }
		inline bool FileInfo::Exists() const { return File::Exists(FullName.to_osstring()); }

		/** Implementation - DirectoryInfo **/

		inline DirectoryInfo::DirectoryInfo(const string& sPath) : FileInfo(sPath) { }

		inline vector<FileInfo> DirectoryInfo::EnumerateFiles() const
		{
			vector<FileInfo> ret;

			#ifdef _WINDOWS

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile((FullName / "*.*").to_osstring().c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				DWORD LastError = GetLastError();
				if (LastError == ERROR_NO_MORE_FILES || LastError == ERROR_FILE_NOT_FOUND) return ret;
				throw IOException();
			}

			for (;;)
			{
				if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					FileInfo fi;
					fi.FullName = FullName / osstring(FindData.cFileName);
					fi.CreationTime = DateTime(FindData.ftCreationTime);
					fi.LastWriteTime = DateTime(FindData.ftLastWriteTime);
					ret.push_back(fi);
				}
			
				if (!FindNextFile(hSearch, &FindData))
				{
					if (GetLastError() == ERROR_NO_MORE_FILES) break;
					throw IOException();
				}
			}

			FindClose(hSearch);

			#else

			DIR *dir;			
			dir = opendir((FullName + (char)Path::DirectorySeparatorChar).c_str());
			if (dir == nullptr) throw DirectoryNotFoundException("Directory '" + FullName + "' was not found.");			
			
			struct dirent *ent;
			while ((ent = readdir (dir)) != nullptr) 
			{
				// if (ent->d_type == DT_DIR) continue;		// It's a directory.
				if (ent->d_type != DT_REG) continue;			// It's a regular file.
				if (IsEqual(ent->d_name, ".") || IsEqual(ent->d_name, "..")) continue;				

				string EntryFullName = FullName + (char)Path::DirectorySeparatorChar + ent->d_name;

				struct stat attrib;         // create a file attribute structure
				if (stat(EntryFullName.c_str(), &attrib) != 0)
				{
					try { Exception::ThrowFromErrno(errno); }
					catch (std::exception& ex) { throw IOException("Unable to retrieve attributes for '" + EntryFullName + "': " + string(ex.what())); }
				}

				FileInfo fi;
				fi.FullName = EntryFullName;
				fi.CreationTime = DateTime(attrib.st_ctime);
				fi.LastWriteTime = DateTime(attrib.st_mtime);
				ret.push_back(fi);
			}
			closedir (dir);			

			#endif
			return ret;
		}

		#ifdef _WINDOWS		// Not yet implemented for linux, but EnumerateFiles() is.
		inline vector<FileInfo> DirectoryInfo::EnumerateFiles(string searchPattern) const
		{
			vector<FileInfo> ret;			

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile((FullName / searchPattern).to_osstring().c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				DWORD LastError = GetLastError();
				if (LastError == ERROR_NO_MORE_FILES || LastError == ERROR_FILE_NOT_FOUND) return ret;
				throw IOException();
			}

			for (;;)
			{
				if (!(FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
				{
					FileInfo fi;
					fi.FullName = FullName / osstring(FindData.cFileName);
					fi.CreationTime = DateTime(FindData.ftCreationTime);
					fi.LastWriteTime = DateTime(FindData.ftLastWriteTime);
					ret.push_back(fi);
				}
			
				if (!FindNextFile(hSearch, &FindData))
				{
					if (GetLastError() == ERROR_NO_MORE_FILES) break;
					throw IOException();
				}
			}

			FindClose(hSearch);						
			
			return ret;
		}
		#endif

		inline vector<DirectoryInfo> DirectoryInfo::EnumerateDirectories() const
		{
			vector<DirectoryInfo> ret;

			#ifdef _WINDOWS

			WIN32_FIND_DATA FindData;
			HANDLE hSearch = FindFirstFile((FullName / "*.*").to_osstring().c_str(), &FindData);
			if (hSearch == INVALID_HANDLE_VALUE)
			{
				DWORD LastError = GetLastError();
				if (LastError == ERROR_NO_MORE_FILES || LastError == ERROR_FILE_NOT_FOUND) return ret;
				throw IOException();
			}

			for (;;)
			{
				if (FindData.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
				{
					if (!IsEqual(FindData.cFileName, L".") && !IsEqual(FindData.cFileName, L".."))
					{
						DirectoryInfo fi;
						fi.FullName = Path::Join(FullName, Path::StripTrailingSeparator(to_string(osstring(FindData.cFileName))));
						fi.CreationTime = DateTime(FindData.ftCreationTime);
						fi.LastWriteTime = DateTime(FindData.ftLastWriteTime);
						ret.push_back(fi);
					}
				}
			
				if (!FindNextFile(hSearch, &FindData))
				{
					if (GetLastError() == ERROR_NO_MORE_FILES) return ret;
					throw IOException();
				}
			}			

			#else

			DIR *dir;			
			dir = opendir((FullName + (char)Path::DirectorySeparatorChar).c_str());
			if (dir == nullptr) throw DirectoryNotFoundException("Directory '" + FullName + "' was not found.");			
			
			struct dirent *ent;
			while ((ent = readdir (dir)) != nullptr) 
			{
				if (ent->d_type != DT_DIR) continue;		// It's a directory.				
				if (IsEqual(ent->d_name, ".") || IsEqual(ent->d_name, "..")) continue;				

				string EntryFullName = FullName + (char)Path::DirectorySeparatorChar + ent->d_name;

				struct stat attrib;         // create a file attribute structure				
				if (stat(EntryFullName.c_str(), &attrib) != 0)
				{
					try { Exception::ThrowFromErrno(errno); }
					catch (std::exception& ex) { throw IOException("Unable to retrieve attributes for '" + EntryFullName + "': " + string(ex.what())); }
				}

				DirectoryInfo fi;
				fi.FullName = EntryFullName;
				fi.CreationTime = DateTime(attrib.st_ctime);
				fi.LastWriteTime = DateTime(attrib.st_mtime);
				ret.push_back(fi);
			}
			closedir (dir);			

			#endif
			return ret;
		}
	}
}

#endif	// __WBFileInfo_h__

//	End of FileInfo.h

