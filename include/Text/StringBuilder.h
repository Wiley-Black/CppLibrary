/////////
//	StringBuilder.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBStringBuilder_h__
#define __WBStringBuilder_h__

#include "../wbFoundation.h"

namespace wb
{
	namespace text
	{
		class StringBuilder
		{
			vector<char> m_buffer;

			/**  Efficiency in changing capacity should probably the responsibility of vector.
			void EfficientResize(size_t To)
			{
				const size_t MB = (size_t)1024 * (size_t)1024;

				if (To < 64) m_buffer.reserve(64);
				else if (To < 256) m_buffer.resize(256);
				else if (To < 4096) m_buffer.resize(4096);
				else if (To < 65536) m_buffer.resize(65536);
				else if (To < MB) m_buffer.resize(To + 65536 - (To % 65536));
				else m_buffer.resize(To + MB - (To % MB));
			}
			*/

		public:
			StringBuilder(string & str)
			{
				m_buffer.resize(str.length());
				CopyMemory(&m_buffer[0], str.c_str(), str.length());
			}
			StringBuilder(int Capacity = 256)
			{
				m_buffer.reserve(Capacity);
			}
			~StringBuilder()
			{
			}
			void Clear() { m_buffer.clear(); }
			void EnsureCapacity(int Capacity)
			{
				if (m_buffer.capacity() < (size_t)Capacity) m_buffer.reserve(Capacity);
			}
			string ToString() { return string(&m_buffer[0], m_buffer.size()); }

			size_t GetLength() { return m_buffer.size(); }

			void SetLength(size_t NewLength)
			{
				m_buffer.resize(NewLength);
			}

			void Append(string txt) 
			{ 
				size_t PrevSize = m_buffer.size();
				m_buffer.resize(m_buffer.size() + txt.length());
				CopyMemory(&m_buffer[PrevSize], txt.c_str(), txt.size());
			}

			void Append(const char *pszText) 
			{ 
				size_t AddSize = strlen(pszText);
				size_t PrevSize = m_buffer.size();
				m_buffer.resize(m_buffer.size() + AddSize);
				CopyMemory(&m_buffer[PrevSize], pszText, AddSize);
			}

			void Append(char ch) 
			{ 				
				size_t PrevSize = m_buffer.size();
				m_buffer.resize(m_buffer.size() + 1);
				m_buffer[PrevSize] = ch;
			}

			void AppendLine(string txt) 
			{ 
				size_t PrevSize = m_buffer.size();
				m_buffer.resize(m_buffer.size() + txt.length() + 1);
				CopyMemory(&m_buffer[PrevSize], txt.c_str(), txt.size());
				m_buffer[m_buffer.size() - 1] = '\n';
			}

			void AppendLine(const char *pszText) 
			{ 
				size_t AddSize = strlen(pszText);
				size_t PrevSize = m_buffer.size();
				m_buffer.resize(m_buffer.size() + AddSize + 1);
				CopyMemory(&m_buffer[PrevSize], pszText, AddSize);
				m_buffer[m_buffer.size() - 1] = '\n';
			}

			void AppendLine() 
			{
				m_buffer.resize(m_buffer.size() + 1);
				m_buffer[m_buffer.size() - 1] = '\n';
			}

			/** Non-.NET Extensions **/

			/// <summary>Retrieves the base address to the start of the string.  Warning: string will not be null-terminated.</summary>
			char* BaseAddress() { return &m_buffer[0]; }
		};
	}
}

#endif	// __WBStringBuilder_h__

//	End of StringBuilder.h

