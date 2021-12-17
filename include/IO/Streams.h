/*	Streams.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBStreams_h__
#define __WBStreams_h__

#include "../Platforms/Platforms.h"
#include <fcntl.h>
#include <errno.h>
#if defined(_MSC_VER)
#include <io.h>
#include <tchar.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <share.h>
#elif defined(_LINUX)
#include <sys/types.h>
#include <unistd.h>
#endif

#include "../Foundation/Exceptions.h"

namespace wb
{
	namespace io
	{		
		enum_class_start(SeekOrigin, int)
		{
			Begin,
			Current,
			End
		}
		enum_class_end(SeekOrigin);		

		class Stream
		{
		public:
			virtual ~Stream() { Close(); }

			virtual bool CanRead() { return false; }
			virtual bool CanWrite() { return false; }
			virtual bool CanSeek() { return false; }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			virtual int ReadByte() { throw NotSupportedException(); }	

			/// <returns>The number of bytes read into the buffer.  This can be less than the requested nLength if not that many bytes are
			/// available, or zero if the end of the stream has been reached.</returns>
			virtual Int64 Read(void *pBuffer, Int64 nLength) 
			{ 
				int ch;
				byte* pDstBuffer = (byte*)pBuffer;
				for (Int64 count = 0; count < nLength; count++)
				{
					ch = ReadByte();
					if (ch < 0) return count;
					*pDstBuffer = (byte)ch; pDstBuffer++;
				}
				return nLength;
			}

			virtual void WriteByte(byte ch) { throw NotSupportedException(); }
			virtual void Write(const void *pBuffer, Int64 nLength) 
			{ 
				byte* pb = (byte *)pBuffer; 
				while (nLength--) WriteByte(*pb++); 
			}

			virtual Int64 GetPosition() const { throw NotSupportedException(); }
			virtual Int64 GetLength() const { throw NotSupportedException(); }
			virtual void Seek(Int64 offset, SeekOrigin origin) { throw NotSupportedException(); }

			virtual void Flush() { }
			virtual void Close() { }
		};				

		inline void StreamToStream(Stream& Source, Stream& Destination)
		{
			byte buffer[4096];
			for (;;)
			{
				Int64 nBytes = Source.Read(buffer, sizeof(buffer));
				if (nBytes == 0) return;			
				Destination.Write(buffer, nBytes);
			}
		}

		inline void StringToStream(const string& Source, Stream& Destination)
		{
			size_t position = 0;
			byte buffer[4096];
			for (;;)
			{
				Int64 nBytes = 0;
				for (; nBytes < 4096 && position < Source.length(); nBytes++, position++) buffer[nBytes] = Source.at(position);
				if (nBytes == 0) break;
				Destination.Write(buffer, nBytes);
			}
		}

		inline string StreamToString(Stream& Source)
		{
			string Destination;
			if (Source.CanSeek())
			{
				Destination.resize(Source.GetLength() - Source.GetPosition());
				Source.Read((void*)Destination.data(), Source.GetLength() - Source.GetPosition());
				return Destination;
			}
			else
			{
				for (;;)
				{
					Int64 CurrentLen = Destination.length();
					Destination.resize(CurrentLen + 4096);
					Int64 nBytes = Source.Read((byte*)Destination.data() + CurrentLen, 4096);
					if (nBytes == 0) {
						Destination.resize(CurrentLen);
						return Destination;
					}
					if (nBytes < 4096) Destination.resize(CurrentLen + nBytes);
				}
			}
		}

		inline string ReadToEnd(Stream& Source)
		{
			string ret;
			byte buffer[4096];
			for (;;)
			{
				Int64 nBytes = Source.Read(buffer, sizeof(buffer) - 1);
				if (nBytes == 0) return ret;
				ret.append((const char *)buffer, (int)nBytes);
			}
		}
	}
}

#endif	// __WBStreams_h__

//	End of Streams.h

