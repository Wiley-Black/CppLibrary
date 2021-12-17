/////////
//  Guid.h
//  Copyright (C) 2001-2014 by Wiley Black
/////////

#ifndef __WBGuid_h__
#define __WBGuid_h__

#include "wbFoundation.h"
#include "Math/Random.h"

#ifndef _WINDOWS		      // Already has GUID support
#define GUID_DEFINED

typedef struct _GUID 
{ 
	unsigned long        Data1;
	unsigned short       Data2;
	unsigned short       Data3;
	unsigned char        Data4[8];
} GUID;

inline bool operator ==(const GUID& a, const GUID& b){
	for( int ii=0; ii < 8; ii++ ) if(a.Data4[ii] != b.Data4[ii]) return false;
	return (a.Data1 == b.Data1 && a.Data2 == b.Data2 && a.Data3 == b.Data3);
}

#endif  // Win32

namespace wb
{
	class Guid : public ::GUID
	{
		static int hex4(char a);
		static bool hex8(char a, char b, unsigned char& value);		
		static bool hex16(char a, char b, char c, char d, unsigned short& value);
		static bool hex32(char a, char b, char c, char d, char e, 
				char f, char g, char h, unsigned long& value);

	public:
		Guid();
		Guid(UInt32 Data1, UInt16 Data2, UInt16 Data3, UInt64 Data4);
		Guid(const ::GUID&);
		Guid& operator=(const GUID&);
		
		static Guid Generate(Random& rng);
		static bool TryParse(string, Guid& Value);
		static Guid Parse(string);
		
		string	ToString() const;
		
		static Guid Zero;
	};

	/////////
	//	Inline Functions
	//

	#if defined(PrimaryModule)
	/*static*/ Guid Guid::Zero(0,0,0,0);
	#endif

	inline Guid::Guid()
	{
		Data1 = Data2 = Data3 = 0;
		ZeroMemory( Data4, sizeof(Data4) );
	}

	inline Guid::Guid(UInt32 _Data1, UInt16 _Data2, UInt16 _Data3, UInt64 _Data4)
	{
		// http://en.wikipedia.org/wiki/Globally_unique_identifier:  A GUID is stored differently than
		// a UUID, but their textual representations are identical.

		Data1 = _Data1;			// Stored in native endian.
		Data2 = _Data2;			// Stored in native endian.
		Data3 = _Data3;			// Stored in native endian.		
		// Data4 is stored in big-endian.
		Data4[0] = (byte)(_Data4 >> 56);
		Data4[1] = (byte)(_Data4 >> 48);
		Data4[2] = (byte)(_Data4 >> 40);
		Data4[3] = (byte)(_Data4 >> 32);
		Data4[4] = (byte)(_Data4 >> 24);
		Data4[5] = (byte)(_Data4 >> 16);
		Data4[6] = (byte)(_Data4 >> 8);
		Data4[7] = (byte)(_Data4);
	}

	inline Guid::Guid(const GUID& cp)
	{
		Data1	= cp.Data1;
		Data2	= cp.Data2;
		Data3	= cp.Data3;
		CopyMemory(Data4, cp.Data4, sizeof(Data4));
	}

	inline Guid& Guid::operator=(const GUID& cp)
	{
		Data1	= cp.Data1;
		Data2	= cp.Data2;
		Data3	= cp.Data3;
		CopyMemory(Data4, cp.Data4, sizeof(Data4));
		return *this;
	}

	inline /*static*/ Guid Guid::Generate(Random& rng)
	{
		Guid ret;
		ret.Data1 = ((UInt32)rng.NextFast() << 16) | (UInt32)rng.NextFast();
		ret.Data2 = rng.NextFast();
		ret.Data3 = rng.NextFast();
		for (int ii=0; ii < 8; ii++) ret.Data4[ii] = (byte)rng.NextFast();
		// Mark as V4 GUID according to wikipedia entry.
		ret.Data4[0] &= ~0x40;
		ret.Data4[0] |= 0x80;		
		ret.Data3 &= ~0xB000;
		ret.Data3 |= 0x40;
		return ret;
	}

	inline string Guid::ToString() const
	{		
		char psz[46];				// I counted 36 characters, but am including extra.
		#if defined(_MSC_VER)
		sprintf_s(psz, "%08x-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", Data1, Data2, Data3, 
			Data4[0], Data4[1], Data4[2], Data4[3], Data4[4], Data4[5], Data4[6], Data4[7]);
		#else
		sprintf(psz, "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x", Data1, Data2, Data3, 
			Data4[0], Data4[1], Data4[2], Data4[3], Data4[4], Data4[5], Data4[6], Data4[7]);
		#endif
		return psz;
	}

	inline int Guid::hex4(char a)
	{
		if( a >= '0' && a <= '9' ) return a - '0';
		return a - 'A' + 0xA;
	}

	inline bool Guid::hex8(char a, char b, unsigned char& value)
	{
		a = toupper(a);
		b = toupper(b);
		if( ((a >= '0' && a <= '9') || (a >= 'A' && a <= 'F'))
		 && ((b >= '0' && b <= '9') || (b >= 'A' && b <= 'F')) ){
			value = (hex4(a) << 4) + hex4(b);
			return true;
		}
		return false;
	}

	inline bool Guid::hex16(char a, char b, char c, char d, unsigned short& value)
	{
		unsigned char ch1, ch2;
		if( !hex8(a,b,ch1) || !hex8(c,d,ch2) ) return false;
		value = (ch1 << 8) + ch2;
		return true;
	}

	inline bool Guid::hex32(char a, char b, char c, char d, char e, char f, char g, char h, unsigned long& value)
	{
		unsigned short w1, w2;
		if( !hex16(a,b,c,d,w1) || !hex16(e,f,g,h,w2) ) return false;
		value = (w1 << 16) + w2;
		return true;
	}

	inline /*static*/ bool Guid::TryParse(string str, Guid& Value)
	{
			// {1D8DA32E-644E-48b6-8E17-D24C1BFD7DDC}
			// 1D8DA32E-644E-48b6-8E17-D24C1BFD7DDC

			// Approach handles both endians.
		
		TrimStart(str);		
		if( str[0] == '{' ) str = str.substr(1);
		TrimStart(str);
		if( !hex32( str[0], str[1], str[2], str[3], str[4], str[5], str[6], str[7], Value.Data1 ) ) return false;
		if( str[8] != '-' ) return false;
		if( !hex16( str[9], str[10], str[11], str[12], Value.Data2 ) ) return false;
		if( str[13] != '-' ) return false;
		if( !hex16( str[14], str[15], str[16], str[17], Value.Data3 ) ) return false;
		if( str[18] != '-' ) return false;
		if( !hex8( str[19], str[20], Value.Data4[0] ) ) return false;
		if( !hex8( str[21], str[22], Value.Data4[1] ) ) return false;
		if( str[23] != '-' ) return false;
		if( !hex8( str[24], str[25], Value.Data4[2] ) ) return false;
		if( !hex8( str[26], str[27], Value.Data4[3] ) ) return false;
		if( !hex8( str[28], str[29], Value.Data4[4] ) ) return false;
		if( !hex8( str[30], str[31], Value.Data4[5] ) ) return false;
		if( !hex8( str[32], str[33], Value.Data4[6] ) ) return false;
		if( !hex8( str[34], str[35], Value.Data4[7] ) ) return false;

		return true;
	}

	inline /*static*/ Guid Guid::Parse(string str)
	{
		Guid ret;
		if (!Guid::TryParse(str, ret)) throw FormatException("Unable to parse GUID value.");
		return ret;
	}
}

#endif  // __WBGuid_h__

//  End of Guid.h

