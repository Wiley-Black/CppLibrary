/////////
//  CRC.h
//  Copyright (C) 1999-2002, 2014 by Wiley Black
/////////
//  Provides CRC-16 and CRC-32 algorithms for an arbitrary polynomial, and the
//	HashedStream template class for computing as a streaming layer.  
//
//	A more efficient implementation of CRC-32 would perform 4-bytes-at-a-time.  See
//	the zlib library for an example.
/////////

#ifndef __WBCRC_h__
#define __WBCRC_h__

#include "../wbFoundation.h"
#include "../Memory Management/Allocation.h"
#include "../Foundation/STL/Memory.h"

namespace wb
{	
	class CRC16Algorithm
	{
		UInt16* m_pTable;
		bool m_PostConditionWithOnesComplement;
		UInt16 m_InitialValue;

	public:
		typedef UInt16 HashType;
		typedef UInt16 ComputationType;

		CRC16Algorithm(UInt16 Polynomial, bool PostConditionWithOnesComplement = false, UInt16 InitialValue = 0xFFFF);
		CRC16Algorithm(const CRC16Algorithm&);
		CRC16Algorithm(CRC16Algorithm&&);
		~CRC16Algorithm();

		void Initialize(ComputationType& Value) const { Value = m_InitialValue; }
		void ComputeHash(const void* pBlock, UInt64 BlockLen, ComputationType& CRC) const;
		void ComputeHash(wb::io::Stream& stream, ComputationType& CRC) const;	
		HashType RetrieveHash(const ComputationType& CRC) const;
	};

	class CRC32Algorithm
	{
		UInt32* m_pTable;
		bool m_PostConditionWithOnesComplement;
		UInt32 m_InitialValue;

	public:
		typedef UInt32 HashType;
		typedef UInt32 ComputationType;

		enum_class_start(CommonPolynomials,UInt32)
		{
			IEEE_802_32bit = 0x04c11db7,
			PkZip = 0xEDB88320
		}
		enum_class_end(CommonPolynomials);

		CRC32Algorithm(UInt32 Polynomial = (UInt32)CommonPolynomials::IEEE_802_32bit, 
			bool PostConditionWithOnesComplement = false, UInt32 InitialValue = 0xFFFFFFFF);
		CRC32Algorithm(const CRC32Algorithm&);
		CRC32Algorithm(CRC32Algorithm&&);
		~CRC32Algorithm();

		void Initialize(ComputationType& Value) const { Value = m_InitialValue; }
		void ComputeHash(const void* pBlock, UInt64 BlockLen, ComputationType& CRC) const;
		void ComputeHash(wb::io::Stream& stream, ComputationType& CRC) const;
		HashType RetrieveHash(const ComputationType& CRC) const;
	};

	namespace io
	{
		using namespace wb::memory;

		template<class HashComputer> class HashedStream : public Stream
		{			
			r_ptr<Stream>							m_pUnderlying;			
			typename HashComputer::ComputationType	m_Calculated;
			const HashComputer*						m_pCalculator;

		public:
			typedef typename HashComputer::HashType HashType;

			/// <summary>Construct a new HashedStream, optionally taking responsibility for closing and deleting the underlying stream.</summary>
			HashedStream(r_ptr<Stream>&& Underlying, const HashComputer& Calculator)
			{
				m_pUnderlying = std::move(Underlying);
				m_pCalculator = &Calculator;
				m_pCalculator->Initialize(m_Calculated);
			}			

			bool CanRead() override { return m_pUnderlying->CanRead(); }
			bool CanWrite() override { return m_pUnderlying->CanWrite(); }
			bool CanSeek() override { return false; }

			/// <summary>Reads one byte from the stream and advances to the next byte position, or returns -1 if at the end of stream.</summary>			
			int ReadByte() override 
			{ 
				int ret = m_pUnderlying->ReadByte();
				if (ret != -1) { byte ch = (byte)ret; m_pCalculator->ComputeHash(&ch, 1, m_Calculated); }
				return ret;
			}	

			/// <returns>The number of bytes read into the buffer.  This can be less than the requested nLength if not that many bytes are
			/// available, or zero if the end of the stream has been reached.</returns>
			Int64 Read(void *pBuffer, Int64 nLength) override
			{ 
				Int64 Bytes = m_pUnderlying->Read(pBuffer, nLength);
				m_pCalculator->ComputeHash(pBuffer, Bytes, m_Calculated);
				return Bytes;				
			}

			void WriteByte(byte ch) override 
			{ 
				m_pCalculator->ComputeHash(&ch, 1, m_Calculated);
				m_pUnderlying->WriteByte(ch);
			}

			void Write(const void *pBuffer, Int64 nLength) override
			{ 
				m_pCalculator->ComputeHash(pBuffer, nLength, m_Calculated);
				m_pUnderlying->Write(pBuffer, nLength);
			}

			Int64 GetPosition() const override { return m_pUnderlying->GetPosition(); }
			Int64 GetLength() const override { return m_pUnderlying->GetLength(); }

			HashType RetrieveHash() { return m_pCalculator->RetrieveHash(m_Calculated); }

			void Close() override { if (m_pUnderlying.IsResponsible()) Close(); }
			void Flush() override { m_pUnderlying->Flush(); }
		};
	}

	/** Implementation - CRC16 **/

	inline CRC16Algorithm::CRC16Algorithm(UInt16 Polynomial, bool PostConditionWithOnesComplement, UInt16 InitialValue)
		: m_PostConditionWithOnesComplement(PostConditionWithOnesComplement),
		m_InitialValue(InitialValue)
	{
		m_pTable = new UInt16 [256];
		for(int ii=0; ii < 256; ii++)
		{			
			UInt16 dw = (UInt16)ii;
			for(int bit=0; bit < 8; bit++)
			{
				if ((dw & 1))
					dw = (dw >> 1) ^ Polynomial;
				else dw >>= 1;
			}
			m_pTable[ii] = (dw & 0xFFFF);
		}
	}

	inline CRC16Algorithm::CRC16Algorithm(const CRC16Algorithm& cp)
		: m_PostConditionWithOnesComplement(cp.m_PostConditionWithOnesComplement),
		m_InitialValue(cp.m_InitialValue)
	{
		m_pTable = new UInt16 [256];
		CopyMemory (m_pTable, cp.m_pTable, sizeof(UInt16) * 256);
	}

	inline CRC16Algorithm::CRC16Algorithm(CRC16Algorithm&& mv)
		: m_PostConditionWithOnesComplement(mv.m_PostConditionWithOnesComplement),
		m_InitialValue(mv.m_InitialValue)
	{
		m_pTable = mv.m_pTable;
		mv.m_pTable = nullptr;
	}

	inline CRC16Algorithm::~CRC16Algorithm()
	{
		delete[] m_pTable;
		m_pTable = nullptr;
	}

	inline void CRC16Algorithm::ComputeHash(const void* pBlock, UInt64 BlockLen, UInt16& CRCValue) const
	{
		byte* pbBlock = (byte*)pBlock;
		while (BlockLen--){
			CRCValue = (CRCValue >> 8) ^ m_pTable[(CRCValue ^ *pbBlock++) & 0xFF];
		}
	}

	inline void CRC16Algorithm::ComputeHash(wb::io::Stream& stream, UInt16& CRCValue) const
	{
		byte buffer[4096];
		for (;;)
		{
			Int64 nBytes = stream.Read(buffer, sizeof(buffer));
			if (nBytes == 0) return;
			ComputeHash(buffer, nBytes, CRCValue);
		}		
	}

	inline CRC16Algorithm::HashType CRC16Algorithm::RetrieveHash(const ComputationType& CRC) const
	{
		if (m_PostConditionWithOnesComplement) return ~CRC; else return CRC;
	}

	/** Implementation - CRC32 **/

	inline CRC32Algorithm::CRC32Algorithm(UInt32 Polynomial, bool PostConditionWithOnesComplement, UInt32 InitialValue)
		: m_PostConditionWithOnesComplement(PostConditionWithOnesComplement),
		m_InitialValue(InitialValue)
	{
		m_pTable = new UInt32 [256];
		for(int ii=0; ii < 256; ii++)
		{			
			UInt32 dw = (UInt32)ii;
			for(int bit=0; bit < 8; bit++)
			{
				if ((dw & 1))
					dw = (dw >> 1) ^ Polynomial;
				else dw >>= 1;
			}
			m_pTable[ii] = (dw & 0xFFFFFFFF);
		}
	}

	inline CRC32Algorithm::CRC32Algorithm(const CRC32Algorithm& cp)
		: m_PostConditionWithOnesComplement(cp.m_PostConditionWithOnesComplement),
		m_InitialValue(cp.m_InitialValue)
	{
		m_pTable = new UInt32 [256];
		CopyMemory (m_pTable, cp.m_pTable, sizeof(UInt32) * 256);
	}

	inline CRC32Algorithm::CRC32Algorithm(CRC32Algorithm&& mv)
		: m_PostConditionWithOnesComplement(mv.m_PostConditionWithOnesComplement),
		m_InitialValue(mv.m_InitialValue)
	{
		m_pTable = mv.m_pTable;
		mv.m_pTable = nullptr;
	}

	inline CRC32Algorithm::~CRC32Algorithm()
	{
		delete[] m_pTable;
		m_pTable = nullptr;
	}

	inline void CRC32Algorithm::ComputeHash(const void* pBlock, UInt64 BlockLen, UInt32& CRCValue) const
	{
		byte* pbBlock = (byte*)pBlock;
		while (BlockLen--){
			CRCValue = (CRCValue >> 8) ^ m_pTable[(CRCValue ^ *pbBlock++) & 0xFF];
		}
	}

	inline void CRC32Algorithm::ComputeHash(wb::io::Stream& stream, UInt32& CRCValue) const
	{
		byte buffer[4096];
		for (;;)
		{
			Int64 nBytes = stream.Read(buffer, sizeof(buffer));
			if (nBytes == 0) return;
			ComputeHash(buffer, nBytes, CRCValue);
		}
	}	

	inline CRC32Algorithm::HashType CRC32Algorithm::RetrieveHash(const ComputationType& CRC) const
	{
		if (m_PostConditionWithOnesComplement) return CRC ^ 0xFFFFFFFF; else return CRC;
	}
}

#endif	// __CrcTables_h__

//  End of CrcTables.h
