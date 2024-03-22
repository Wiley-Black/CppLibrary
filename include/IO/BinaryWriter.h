/*	BinaryWriter.h
	Copyright (C) 2024 by Wiley Black (TheWiley@gmail.com)
*/

#ifndef __WBBinaryWriter_h__
#define __WBBinaryWriter_h__

#include "wbFoundation.h"
#include "../IO/Streams.h"
#include "../Memory Management/Allocation.h"
#include "../IO/FileStream.h"

namespace wb
{
	namespace io
	{				
		class BinaryWriter			
		{
		protected:
			bool						m_LittleEndian;
			bool						m_SwapEndian;
			memory::r_ptr<Stream>		m_pStream;					

			void WriteToEndian(void* pData, UInt64 length);

		public:
			BinaryWriter(memory::r_ptr<Stream>&& rpStream, bool little_endian = true)
				: 												
				m_LittleEndian(little_endian),
				m_pStream(std::move(rpStream))
			{ 
				m_SwapEndian = (IsLittleEndian() != m_LittleEndian);
			}

			BinaryWriter(const Path& path, bool little_endian = true, bool append = true)
				:				
				m_LittleEndian(little_endian),
				m_pStream(memory::r_ptr<Stream>::responsible(new FileStream(
					path.to_osstring(), append ? FileMode::Append : FileMode::Create, 
					FileAccess::ReadWrite, FileShare::Read)))
			{ 
				m_SwapEndian = (IsLittleEndian() != m_LittleEndian);
			}

			// Write() routines that do not perform endian translation:
			void Write(void* pData, UInt64 length);
			void Write(Stream& from_source);

			// Write() routines that handle any necessary endian swapping:
			void Write(UInt64 value) { WriteToEndian(&value, sizeof(UInt64)); }
			void Write(UInt32 value) { WriteToEndian(&value, sizeof(UInt32)); }
			void Write(UInt16 value) { WriteToEndian(&value, sizeof(UInt16)); }
			void Write(UInt8 value) { Write(&value, sizeof(UInt8)); }
			void Write(Int64 value) { WriteToEndian(&value, sizeof(Int64)); }
			void Write(Int32 value) { WriteToEndian(&value, sizeof(Int32)); }
			void Write(Int16 value) { WriteToEndian(&value, sizeof(Int16)); }
			void Write(Int8 value) { Write(&value, sizeof(Int8)); }
			void Write(float value) { WriteToEndian(&value, sizeof(float)); }
			void Write(double value) { WriteToEndian(&value, sizeof(double)); }			

			// Output stream utility functions:
			void Seek(Int64 offset, wb::io::SeekOrigin origin);
			Int64 GetPosition();
			void Flush();
		};

		/** Implementation **/

		inline void BinaryWriter::Write(void* pData, UInt64 length)
		{
			m_pStream->Write(pData, length);
		}

		inline void BinaryWriter::WriteToEndian(void* pData, UInt64 length)
		{
			if (length == 0) return;
			if (m_SwapEndian)
			{
				byte* pRevData = ((byte*)pData) + (length - 1);
				for (UInt64 ii = 0; ii < length; ii++, pRevData--)
					m_pStream->WriteByte(*pRevData);
			}
			else
				m_pStream->Write(pData, length);
		}

		inline void BinaryWriter::Write(Stream& from_source)
		{			
			byte buffer[4096];
			for (;;)
			{
				Int64 nBytes = from_source.Read(buffer, sizeof(buffer) - 1);
				if (nBytes == 0) return;
				m_pStream->Write(buffer, nBytes);
			}
		}

		inline void BinaryWriter::Seek(Int64 offset, wb::io::SeekOrigin origin)
		{
			m_pStream->Seek(offset, origin);
		}

		inline Int64 BinaryWriter::GetPosition()
		{
			return m_pStream->GetPosition();
		}

		inline void BinaryWriter::Flush()
		{
			m_pStream->Flush();
		}
	}
}

#endif	// __WBBinaryWriter_h__

//	End of BinaryWriter.h

