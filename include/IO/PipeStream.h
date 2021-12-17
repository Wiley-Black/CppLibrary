/*	PipeStream.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBPipeStream_h__
#define __WBPipeStream_h__

#include "../IO/Streams.h"

#ifdef _WINDOWS			// TODO: Add Linux support

namespace wb
{
	namespace io
	{		
		enum_class_start(PipeDirection, int)
		{
			/// <summary>Specifies that the pipe direction is in.</summary>
			In,

			/// <summary>Specifies that the pipe direction is two-way.</summary>			
			InOut,

			/// <summary>Specifies that the pipe direction is out.</summary>
			Out
		}
		enum_class_end(PipeDirection);
	
		class PipeStream : public Stream
		{
		protected:
			#ifdef _WINDOWS
			HANDLE m_hReadPipe;
			HANDLE m_hWritePipe;
			#else
			int	m_Handle;
			#endif

			bool m_bCanRead;
			bool m_bCanWrite;						

		public:			

			PipeStream()
			{
				#if defined(_WINDOWS)
				m_hReadPipe = INVALID_HANDLE_VALUE;
				m_hWritePipe = INVALID_HANDLE_VALUE;
				#else
				m_Handle = -1;
				#endif
				m_bCanRead = m_bCanWrite = false;
			}

			PipeStream(PipeDirection direction, int bufferSize = 0)
			{
				#if defined(_WINDOWS)
				m_hReadPipe = INVALID_HANDLE_VALUE;
				m_hWritePipe = INVALID_HANDLE_VALUE;
				#else
				m_Handle = -1;
				#endif

				m_bCanRead = m_bCanWrite = false;

				SECURITY_ATTRIBUTES saAttr; 
				saAttr.nLength = sizeof(SECURITY_ATTRIBUTES); 
				saAttr.bInheritHandle = TRUE; 
				saAttr.lpSecurityDescriptor = NULL; 				

				if (!::CreatePipe(&m_hReadPipe, &m_hWritePipe, &saAttr, bufferSize) ) 
					Exception::ThrowFromWin32(::GetLastError());

				switch (direction)
				{
				case PipeDirection::In:
					if ( !::SetHandleInformation(m_hReadPipe, HANDLE_FLAG_INHERIT, 0) )
						Exception::ThrowFromWin32(::GetLastError());
					m_bCanWrite = false;
					break;
				case PipeDirection::Out:
					if ( !::SetHandleInformation(m_hWritePipe, HANDLE_FLAG_INHERIT, 0) )
						Exception::ThrowFromWin32(::GetLastError());
					m_bCanRead = false;
					break;
				default:
					m_bCanRead = m_bCanWrite = true;
				}				
			}			

			PipeStream(const PipeStream& cp) { throw Exception("Cannot copy a PipeStream object."); }
			PipeStream& operator=(const PipeStream& cp) { throw Exception("Cannot copy a PipeStream object."); }
			PipeStream(PipeStream&& mv) { operator=(mv); }
			PipeStream& operator=(PipeStream&& mv) 
			{
				m_hReadPipe = mv.m_hReadPipe;
				m_hWritePipe = mv.m_hWritePipe;
				m_bCanRead = mv.m_bCanRead;
				m_bCanWrite = mv.m_bCanWrite;
				#if defined(_WINDOWS)
				mv.m_hReadPipe = INVALID_HANDLE_VALUE;
				mv.m_hWritePipe = INVALID_HANDLE_VALUE;
				m_bCanRead = m_bCanWrite = false;
				#else
				mv.m_Handle = -1;
				m_bCanRead = m_bCanWrite = false;
				#endif
				return *this;
			}

			~PipeStream() { Close(); }			

			bool CanRead() override { return m_bCanRead; }
			bool CanWrite() override { return m_bCanWrite; }
			bool CanSeek() override { return false; }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			int ReadByte() override 
			{ 				
				byte ch;
				#if defined(_WINDOWS)
				DWORD count;
				if (!::PeekNamedPipe(m_hReadPipe, NULL, 0, NULL, &count, NULL)) Exception::ThrowFromWin32(::GetLastError());
				if (count < 1) return -1;
				if (!::ReadFile(m_hReadPipe, &ch, 1, &count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
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
					if (!::PeekNamedPipe(m_hReadPipe, NULL, 0, NULL, &block_count, NULL)) Exception::ThrowFromWin32(::GetLastError());
					if (block_count < Int32_MaxValue) break;
					if (!::ReadFile(m_hReadPipe, pDstBuffer + count, Int32_MaxValue, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());					
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
				if (!::PeekNamedPipe(m_hReadPipe, NULL, 0, NULL, &block_count, NULL)) Exception::ThrowFromWin32(::GetLastError());
				if (nLength > block_count) nLength = block_count;
				if (block_count == 0) return count;
				if (!::ReadFile(m_hReadPipe, pDstBuffer + count, (Int32)nLength, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());									
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
				if (!::WriteFile(m_hWritePipe, &ch, 1, &count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
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
					if (!::WriteFile(m_hWritePipe, pBuffer, Int32_MaxValue, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
					if (block_count != Int32_MaxValue) throw IOException();
					#else
					if (write(m_Handle, pBuffer, Int32_MaxValue) != nLength) Exception::ThrowFromErrno(errno);
					#endif
					pBuffer = ((byte *)pBuffer) + Int32_MaxValue;
					nLength -= Int32_MaxValue;
				}
				#if defined(_WINDOWS)
				DWORD block_count;
				if (!::WriteFile(m_hWritePipe, pBuffer, (Int32)nLength, &block_count, nullptr)) Exception::ThrowFromWin32(::GetLastError());
				if (block_count != (UInt32)nLength) throw IOException();
				#else
				if (write(m_Handle, pBuffer, (Int32)nLength) != nLength) Exception::ThrowFromErrno(errno);
				#endif
			}

			Int64 GetPosition() const override { throw NotSupportedException("Pipes do not have a position."); }
			Int64 GetLength() const override { throw NotSupportedException("Pipes do not have a fixed length."); }				
			void Seek(Int64 offset, SeekOrigin origin) override { throw NotSupportedException("Cannot seek within a PipeStream."); }							

			void Close()
			{
				Flush();
				#if defined(_WINDOWS)
				if (m_hReadPipe != INVALID_HANDLE_VALUE) { ::CloseHandle(m_hReadPipe); m_hReadPipe = INVALID_HANDLE_VALUE; }
				if (m_hWritePipe != INVALID_HANDLE_VALUE) { ::CloseHandle(m_hWritePipe); m_hWritePipe = INVALID_HANDLE_VALUE; }
				#else
				if (m_Handle != -1) { close(m_Handle); m_Handle = -1; }
				#endif
				m_bCanRead = m_bCanWrite = false;
			}

			#if defined(_WINDOWS)
			HANDLE GetReadHandle() { return m_hReadPipe; }
			HANDLE GetWriteHandle() { return m_hWritePipe; }
			#endif
		};
	}
}

#endif

#endif	// __PipeStream_h__

//	End of PipeStream.h

