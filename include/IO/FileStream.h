/*	FileStream.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

// Include wbFoundation.h ahead in order to enforce the order of header inclusion.
#include "../wbFoundation.h"

#ifndef __WBFileStream_h__
#define __WBFileStream_h__

#include "Streams.h"
#include "Path.h"

namespace wb
{
	namespace io
	{		
		enum_class_start(FileMode, int)
		{
			/// <summary>Opens the file if it exists and seeks to the end of the file, or creates a new file. 
			/// FileMode::Append can be used only in conjunction with FileAccess::Write.</summary>
			Append,

			/// <summary>Specifies that the operating system should create a new file. If the file already exists, it will be overwritten. 
			/// FileMode.Create is equivalent to requesting that if the file does not exist, use CreateNew; otherwise, use Truncate.</summary>			
			Create,

			/// <summary>Specifies that the operating system should create a new file.</summary>
			CreateNew,

			/// <summary>Specifies that the operating system should open an existing file. </summary>
			Open,

			/// <summary>Specifies that the operating system should open a file if it exists; otherwise, a new file should be created.</summary>
			OpenOrCreate,

			/// <summary>Specifies that the operating system should open an existing file. When the file is opened, it should be truncated so that its size is zero bytes.</summary>
			Truncate
		}
		enum_class_end(FileMode);		

		enum_class_start(FileAccess,int)
		{
			Read,
			ReadWrite,
			Write
		}
		enum_class_end(FileAccess);

		enum_class_start(FileShare,int)
		{
			None,
			Read,
			ReadWrite,
			Write
		}
		enum_class_end(FileShare);

		class FileStream : public Stream
		{
		protected:
			#ifdef _WINDOWS
			HANDLE m_Handle;
			#else
			int	m_Handle;
			#endif

			bool m_bCanRead;
			bool m_bCanWrite;			

			#if !defined(_WINDOWS)
			void Open(const char *pszFilename, int opflags)
			{
				if (pszFilename == nullptr) throw ArgumentException(S("Null value for filename."));

				// In an "interesting" design decision, Linux defines O_RDONLY as 0, perhaps suggesting
				// that a file can always be read even though O_RDWR is 0x2.  In any case, 0 is a valid
				// opflag saying open a file for reading, so we can't really verify the read flag.
				if (opflags & O_RDWR) { m_bCanRead = true; m_bCanWrite = true; }
				else if (opflags & O_WRONLY) { m_bCanRead = false; m_bCanWrite = true; }
				else { m_bCanRead = true; m_bCanWrite = false; }
				//else throw ArgumentException(string("Invalid read/write opflags (") + to_string(opflags) + ").");
				
				m_Handle = open(pszFilename, opflags | O_LARGEFILE, S_IRUSR | S_IWUSR | S_IRGRP | S_IWGRP);
				if (m_Handle >= 0) return;			// Done, successful.
				
				m_Handle = -1;
				switch (errno)
				{
				case EACCES: throw UnauthorizedAccessException(S("Unauthorized access."));
				case EEXIST: throw IOException(S("File already exists."));
				case EINVAL: throw Exception(S("An internal error occurred while opening file."));
				case EMFILE: throw IOException(S("No more file descriptors available."));
				case ENOENT: throw FileNotFoundException(S("File or path not found."));
				default: Exception::ThrowFromErrno(errno);
				}
			}
			#endif						

			void Init()
			{
				#if defined(_WINDOWS)
				m_Handle = INVALID_HANDLE_VALUE;
				#else
				m_Handle = -1;
				#endif

				m_bCanRead = m_bCanWrite = false;				
			}

		public:			

			FileStream()
			{
				Init();				
			}

			FileStream(const char* pszFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				Init();
				Open(wb::to_osstring(pszFilename).c_str(), mode, access, share);
			}

			FileStream(const wchar_t* pszFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				Init();
				Open(wb::to_osstring(pszFilename).c_str(), mode, access, share);
			}

			FileStream(const string& sFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				Init();
				Open(wb::to_osstring(sFilename).c_str(), mode, access, share);
			}

			FileStream(const wstring& sFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				Init();
				Open(to_osstring(sFilename).c_str(), mode, access, share);
			}

			FileStream(const Path& Filename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				Init();
				Open(Filename.to_osstring().c_str(), mode, access, share);
			}

			#if defined(_WINDOWS)
			static FileStream FromHandle(HANDLE hFile, bool CanRead, bool CanWrite)
			{
				FileStream ret;
				ret.m_Handle = hFile;
				ret.m_bCanRead = CanRead;
				ret.m_bCanWrite = CanWrite;
				return ret;
			}
			#endif

			FileStream(const FileStream& cp) { throw Exception("Cannot copy a FileStream object."); }
			FileStream& operator=(const FileStream& cp) { throw Exception("Cannot copy a FileStream object."); }
			FileStream(FileStream&& mv) noexcept { operator=(mv); }
			FileStream& operator=(FileStream&& mv) noexcept
			{
				m_Handle = mv.m_Handle;
				m_bCanRead = mv.m_bCanRead;
				m_bCanWrite = mv.m_bCanWrite;
				#if defined(_WINDOWS)
				mv.m_Handle = INVALID_HANDLE_VALUE;
				m_bCanRead = m_bCanWrite = false;
				#else
				mv.m_Handle = -1;
				m_bCanRead = m_bCanWrite = false;
				#endif
				return *this;
			}

			~FileStream() { Close(); }

			#if defined(_WINDOWS)
			void Open(const TCHAR *pszFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				if (pszFilename == nullptr) throw ArgumentException(S("Null value for filename."));

				DWORD dwDesiredAccess = 0;				
				switch (access)
				{
				case FileAccess::Read: m_bCanRead = true; dwDesiredAccess |= GENERIC_READ; break;
				case FileAccess::Write: m_bCanWrite = true; dwDesiredAccess |= GENERIC_WRITE; break;
				case FileAccess::ReadWrite: m_bCanRead = m_bCanWrite = true; dwDesiredAccess |= GENERIC_READ | GENERIC_WRITE; break;
				default: throw ArgumentException(S("Invalid file access flag."));
				}

				DWORD dwCreationDisposition;
				switch (mode)
				{
				case FileMode::Append: 
					if (!m_bCanWrite) throw ArgumentException(S("FileMode::Append requires FileAccess::Write or FileAccess::ReadWrite."));
					dwCreationDisposition = OPEN_ALWAYS;
					break;
				case FileMode::Create: 
					if (!m_bCanWrite) throw ArgumentException(S("FileMode::Create requires FileAccess::Write or FileAccess::ReadWrite."));
					dwCreationDisposition = CREATE_ALWAYS;
					break;
				case FileMode::CreateNew: dwCreationDisposition = CREATE_NEW; break;
				case FileMode::Open: dwCreationDisposition = OPEN_EXISTING; break;
				case FileMode::OpenOrCreate: dwCreationDisposition = OPEN_ALWAYS; break;
				case FileMode::Truncate: dwCreationDisposition = TRUNCATE_EXISTING; break;
				default: throw ArgumentException(S("Invalid file mode flag."));
				}
				
				DWORD dwShareMode = 0;		// No option for FILE_SHARE_DELETE (and rename) presently available here.				
				switch (share)
				{
				case FileShare::None: dwShareMode = 0; break;
				case FileShare::Read: dwShareMode |= FILE_SHARE_READ; break;
				case FileShare::Write: dwShareMode |= FILE_SHARE_WRITE; break;
				case FileShare::ReadWrite: dwShareMode |= FILE_SHARE_READ | FILE_SHARE_WRITE; break;
				default: throw ArgumentException(S("Invalid share mode flag."));
				}				

				m_Handle = ::CreateFile(pszFilename, dwDesiredAccess, dwShareMode, nullptr, dwCreationDisposition, FILE_ATTRIBUTE_NORMAL, nullptr);
				if (m_Handle == INVALID_HANDLE_VALUE) Exception::ThrowFromWin32(::GetLastError());

				if (mode == FileMode::Append)
					::SetFilePointer(m_Handle, 0, NULL, FILE_END);
			}
			#else
			void Open(const char *pszFilename, FileMode mode, FileAccess access = FileAccess::ReadWrite, FileShare share = FileShare::Read)
			{
				if (m_Handle >= 0) throw NotSupportedException("File '" + string(pszFilename) + "' is already open!");				

				int opflags = 0;
				switch (mode)
				{
				case FileMode::Append: opflags = O_CREAT; break;			// O_APPEND on Linux is a bit different than what FileStream really wants to achieve.  We'll seek below.
				case FileMode::Create: opflags = O_CREAT | O_TRUNC; break;
				case FileMode::CreateNew: opflags = O_CREAT | O_EXCL; break;
				case FileMode::Open: opflags = 0; break;
				case FileMode::OpenOrCreate: opflags = O_CREAT; break;
				case FileMode::Truncate: opflags = O_TRUNC; break;
				default: throw ArgumentException(S("Invalid file mode flag."));
				}

				switch (access)
				{
				case FileAccess::Read: opflags |= O_RDONLY; break;
				case FileAccess::ReadWrite: opflags |= O_RDWR; break;
				case FileAccess::Write: opflags |= O_WRONLY; break;
				default: throw ArgumentException(S("Invalid file access flag."));
				}

				// Note: shflags are ignored on Linux.
				#ifndef _LINUX
				int shflags;
				switch (share)
				{
				case FileShare::None: shflags = _SH_DENYRW; break;
				case FileShare::Read: shflags = _SH_DENYWR; break;
				case FileShare::Write: shflags = _SH_DENYRD; break;
				case FileShare::ReadWrite: shflags = _SH_DENYNO; break;
				default: throw ArgumentException(S("Invalid share mode flag."));
				}
				#endif

				Open(pszFilename, opflags);

				if (mode == FileMode::Append) Seek(0, SeekOrigin::End);
			}
			#endif

			bool CanRead() override { return m_bCanRead; }
			bool CanWrite() override { return m_bCanWrite; }
			bool CanSeek() override { return true; }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			int ReadByte() override 
			{ 
				byte ch;
				#if defined(_WINDOWS)
				DWORD count;
				if (!::ReadFile(m_Handle, &ch, 1, &count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
				#else
				int count = read(m_Handle, &ch, 1);
				#endif
				if (count == 1) return ch;
				if (count == 0) return -1;
				#if !defined(_WINDOWS)
				Exception::ThrowFromErrno(errno);
				#endif
				throw IOException();
			}
			Int64 Read(void* pBuffer, Int64 nLength) override 
			{	
				Int64 count = 0;
				byte* pDstBuffer = (byte *)pBuffer;
				while (nLength > Int32_MaxValue)
				{
					#if defined(_WINDOWS)
					DWORD block_count;
					if (!::ReadFile(m_Handle, pDstBuffer + count, Int32_MaxValue, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());					
					#else					
					Int32 block_count = read(m_Handle, pDstBuffer + count, Int32_MaxValue);
					if (block_count < 0) Exception::ThrowFromErrno(errno);
					#endif					
					count += block_count;
					if (block_count < Int32_MaxValue) return count;
					nLength -= block_count;
				}

				#if defined(_WINDOWS)
				DWORD block_count;
				if (!::ReadFile(m_Handle, pDstBuffer + count, (Int32)nLength, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());									
				return (Int64)block_count + count;
				#else				
				Int32 block_count = read(m_Handle, pDstBuffer + count, (Int32)nLength);				
				if (block_count >= 0) return (Int64)block_count + count;
				Exception::ThrowFromErrno(errno);
				#endif
			}

			void WriteByte(byte ch) override 
			{
				#if defined(_WINDOWS)
				DWORD count;
				if (!::WriteFile(m_Handle, &ch, 1, &count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
				if (count != 1) throw IOException();
				#else
				if (write(m_Handle, &ch, 1) != 1) Exception::ThrowFromErrno(errno);
				#endif
			}			
			void Write(const void *pBuffer, Int64 nLength) override 
			{ 
				while (nLength > Int32_MaxValue)
				{
					#if defined(_WINDOWS)
					DWORD block_count;
					if (!::WriteFile(m_Handle, pBuffer, Int32_MaxValue, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
					if (block_count != Int32_MaxValue) throw IOException();
					#else
					if (write(m_Handle, pBuffer, Int32_MaxValue) != nLength) Exception::ThrowFromErrno(errno);
					#endif
					pBuffer = ((byte *)pBuffer) + Int32_MaxValue;
					nLength -= Int32_MaxValue;
				}
				#if defined(_WINDOWS)
				DWORD block_count;
				if (!::WriteFile(m_Handle, pBuffer, (Int32)nLength, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
				if (block_count != (UInt32)nLength) throw IOException();
				#else
				if (write(m_Handle, pBuffer, (Int32)nLength) != nLength) Exception::ThrowFromErrno(errno);
				#endif
			}

			Int64 GetPosition() const override { 
				#if defined(_WINDOWS)				
				LARGE_INTEGER zero; zero.QuadPart = 0;
				LARGE_INTEGER fp;
				if (!::SetFilePointerEx(m_Handle, zero, &fp, FILE_CURRENT)) Exception::ThrowFromWin32(::GetLastError());
				return fp.QuadPart;
				#else
				Int64 fp = lseek64(m_Handle, 0, SEEK_CUR);
				if (fp < 0) throw IOException();
				return fp;
				#endif
			}

			Int64 GetLength() const override { 
				#if defined(_WINDOWS)
				LARGE_INTEGER size;
				if (!::GetFileSizeEx(m_Handle, &size)) Exception::ThrowFromWin32(::GetLastError());
				return size.QuadPart;
				#else
				Int64 fp = lseek64(m_Handle, 0, SEEK_CUR);
				if (fp < 0) throw IOException();
				Int64 length = lseek64(m_Handle, 0, SEEK_END);
				if (length < 0) throw IOException();
				if (lseek64(m_Handle, fp, SEEK_SET) < 0) throw IOException();
				return length;
				#endif
			}

			void Seek(Int64 offset, SeekOrigin origin) override 
			{
				#if defined(_WINDOWS)
				LARGE_INTEGER liOffset; liOffset.QuadPart = offset;				
				
				switch (origin)
				{				
				case SeekOrigin::Begin: if (!::SetFilePointerEx(m_Handle, liOffset, nullptr, FILE_BEGIN)) Exception::ThrowFromWin32(::GetLastError()); return;
				case SeekOrigin::Current:  if (!::SetFilePointerEx(m_Handle, liOffset, nullptr, FILE_CURRENT)) Exception::ThrowFromWin32(::GetLastError()); return;					
				case SeekOrigin::End: if (!::SetFilePointerEx(m_Handle, liOffset, nullptr, FILE_END)) Exception::ThrowFromWin32(::GetLastError()); return;
				default: throw ArgumentException(S("Invalid origin."));
				}
				#else				
				Int64 result;
				switch (origin)
				{				
				case SeekOrigin::Begin: result = lseek64(m_Handle, offset, SEEK_SET); break;
				case SeekOrigin::Current: result = lseek64(m_Handle, offset, SEEK_CUR); break;
				case SeekOrigin::End: result = lseek64(m_Handle, offset, SEEK_END); break;				
				default: throw ArgumentException(S("Invalid origin."));
				}
				if (result < 0) Exception::ThrowFromErrno(errno);
				#endif
			}

			void Close()
			{
				Flush();
				#if defined(_WINDOWS)
				if (m_Handle != INVALID_HANDLE_VALUE) { ::CloseHandle(m_Handle); m_Handle = INVALID_HANDLE_VALUE; }
				#else
				if (m_Handle != -1) { close(m_Handle); m_Handle = -1; }
				#endif
				m_bCanRead = m_bCanWrite = false;
			}

			#if defined(_WINDOWS)
			HANDLE GetHandle() { return m_Handle; }
			#endif			
		};
	}
}

#endif	// __FileStream_h__

//	End of FileStream.h

