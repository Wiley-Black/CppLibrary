/*	StreamWriter.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBStreamWriter_h__
#define __WBStreamWriter_h__

#include <sstream>

#include "wbFoundation.h"
#include "../IO/Streams.h"
#include "../Memory Management/Allocation.h"
#include "../IO/FileStream.h"

namespace wb
{
	namespace io
	{		
		/// StreamWriter, similar to the C# class.  Presently provides only a "pass-thru" encoding, whereas a proper implementation
		/// will give the encoding more thought as the C# class does.
		class StreamWriter 
			// : public std::ostream, public std::stringbuf
		{
			memory::r_ptr<Stream>		m_pStream;

		protected:

			/**
			#ifdef UseSTL
			class StreamWriterBuffer : public std::stringbuf 
			{
				StreamWriter&	m_Parent;

			public:
				StreamWriterBuffer(StreamWriter& Parent) : m_Parent(Parent) { }
				~StreamWriterBuffer() { pubsync(); }
				int sync() { m_Parent.Write(str()); str(""); return 0; }
			};
			StreamWriterBuffer	m_Buffer;
			#endif
			**/
			//int sync() { Write(str()); str(""); return 0; }

		public:
			StreamWriter(memory::r_ptr<Stream>&& rpStream)
				: 								
				//std::ostream(this),
				m_pStream(std::move(rpStream))
			{ }

			/*
			StreamWriter(string path, bool append = true)
				: 								
				std::ostream(this),
				m_pStream(memory::r_ptr<Stream>::responsible(new FileStream(path, append ? FileMode::Append : FileMode::Create)))
			{ }

			StreamWriter(wstring path, bool append = true)
				:
				std::ostream(this),
				m_pStream(memory::r_ptr<Stream>::responsible(new FileStream(path, append ? FileMode::Append : FileMode::Create)))
			{ }
			*/

			StreamWriter(const Path& path, bool append = true)
				:
				//std::ostream(this),
				m_pStream(memory::r_ptr<Stream>::responsible(new FileStream(path.to_osstring(), append ? FileMode::Append : FileMode::Create)))
			{ }

			void Write(const string&);
			void WriteLine(const string& = "");
			void Flush();
		};

		/** Implementation **/

		inline void StreamWriter::Write(const string& str)
		{
			m_pStream->Write(str.c_str(), str.length());
		}

		inline void StreamWriter::WriteLine(const string& str)
		{
			static const char* pszEOL = "\r\n";
			m_pStream->Write(str.c_str(), str.length());
			m_pStream->Write(pszEOL, 2);
		}

		inline void StreamWriter::Flush()
		{
			m_pStream->Flush();
		}
	}
}

#endif	// __WBStreamWriter_h__

//	End of StreamWriter.h

