/////////
//  IP4Address.h
//  Copyright (C) 2000-2002, 2014 by Wiley Black
/////////

#ifndef __Socket_IP4Address_h__
#define __Socket_IP4Address_h__

#include "wbFoundation.h"
#include "Sockets.h"
#include <assert.h>

#if !defined(_WINDOWS)
#include <netinet/in.h>
#endif

#undef SetPort

namespace wb
{
	namespace net
	{
		namespace sockets
		{			
			/** IP4Address **/

			class IP4Address
			{
			public:
				sockaddr	m_sockaddr;				// Stored in network byte order.
				enum	{ AddrSize = 16 };			// Portion (front) in bytes of sockaddr which is used in this class.

			public:

				enum	{ ANY = 0 };							// Applies to both IP addresses and Port numbers.

				IP4Address();								// Creates an IP Address of ANY:ANY (assigned by service provider).
				IP4Address(const IP4Address& cp);
				IP4Address(IP4Address&& mv) noexcept;
				IP4Address(sockaddr& addr);					// 'addr' should be in network byte order.
				IP4Address(byte ip1, byte ip2, byte ip3, byte ip4);		// IP: Dotted notation order.  Port: Host byte order.
				IP4Address(byte ip1, byte ip2, byte ip3, byte ip4, UInt16 nPort);// IP: Dotted notation order.  Port: Host byte order.
				IP4Address(UInt32 dwIP, UInt16 nPort = ANY);	// Specify both values in host byte order.								

				bool Set(const IP4Address& cp){ *this = cp; return true; }				
				void Set(byte ip1, byte ip2, byte ip3, byte ip4, UInt16 nPort);
				void SetIP(UInt32 ip);
				void SetPort(UInt16 port);

				IP4Address& operator=(const sockaddr& cp);
				IP4Address& operator=(const IP4Address& cp);
				IP4Address& operator=(IP4Address&& mv) noexcept;
				bool operator==(const IP4Address& cp) const;
				bool operator!=(const IP4Address& cp) const;				

				UInt16	GetPort() const;						// Returns the port number in host byte order.
				UInt32	GetIP() const;							// Returns the IP address in host byte order.
				void	GetIP( byte& b1, byte& b2, byte& b3, byte& b4 ) const;		// Returns the IP address in dotted notation order.
				
				string				ToString(bool ShowPort = false) const;
				static bool			TryParse(const string& str, IP4Address& Value, UInt16 nDefaultPort = 0);
				static IP4Address	Parse(const string& str, UInt16 nDefaultPort = 0);

				static IP4Address AnyAddress;
			};
			
			/** Inline functions **/

			#ifdef PrimaryModule
			IP4Address IP4Address::AnyAddress = IP4Address(ANY,ANY);
			#endif

			inline IP4Address::IP4Address()
			{
				ZeroMemory(&m_sockaddr, sizeof(m_sockaddr));								
				m_sockaddr.sa_family = AF_INET;					// Internet Address Family				
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;
				in.sin_addr.s_addr = INADDR_ANY;				// IP Address defaults to "ANY" (assigned by service provider)
				in.sin_port = 0;								// Port Address defaults to 0 (assigned by service provider)
			}

			inline IP4Address::IP4Address(const IP4Address& cp)
			{				
				CopyMemory(&m_sockaddr, &cp.m_sockaddr, sizeof(m_sockaddr));
			}

			inline IP4Address::IP4Address(IP4Address&& mv) noexcept
			{
				CopyMemory(&m_sockaddr, &mv.m_sockaddr, sizeof(m_sockaddr));
			}

			inline IP4Address::IP4Address(sockaddr& addr)
			{
				if (addr.sa_family != AF_INET) throw FormatException("Unable to store address of non-IPv4 format in IPv4 container.");
				CopyMemory(&m_sockaddr, &addr, sizeof(m_sockaddr));
			}

			inline IP4Address::IP4Address(byte ip1, byte ip2, byte ip3, byte ip4)
			{
				ZeroMemory(&m_sockaddr, sizeof(m_sockaddr));
				m_sockaddr.sa_family = AF_INET;
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;
				in.sin_port = 0;								// Port Address defaults to 0 (assigned by service provider)
				in.sin_addr.s_addr = htonl(MakeU32(ip4, ip3, ip2, ip1));
			}

			inline IP4Address::IP4Address(byte ip1, byte ip2, byte ip3, byte ip4, UInt16 nPort)
			{
				ZeroMemory(&m_sockaddr, sizeof(m_sockaddr));
				m_sockaddr.sa_family = AF_INET;
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;
				in.sin_port = htons(nPort);
				in.sin_addr.s_addr = htonl(MakeU32(ip4, ip3, ip2, ip1));
			}

			inline IP4Address::IP4Address(UInt32 nIP, UInt16 nPort)
			{
				ZeroMemory( &m_sockaddr, sizeof(m_sockaddr) );
				m_sockaddr.sa_family = AF_INET;
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;				
				in.sin_port = htons(nPort);
				in.sin_addr.s_addr = htonl(nIP);
			}			

			inline void IP4Address::Set(byte ip1, byte ip2, byte ip3, byte ip4, UInt16 nPort)
			{
				ZeroMemory( &m_sockaddr, sizeof(m_sockaddr) );
				m_sockaddr.sa_family = AF_INET;
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;
				in.sin_port = htons( nPort );				
				in.sin_addr.s_addr = htonl(MakeU32(ip4, ip3, ip2, ip1));
			}

			inline void IP4Address::SetIP(UInt32 nIP)
			{	
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;				
				in.sin_addr.s_addr = htonl(nIP);
			}

			inline void IP4Address::SetPort(UInt16 nPort)
			{	
				sockaddr_in& in = (sockaddr_in&)m_sockaddr;
				in.sin_port = htons(nPort);
			}

			inline IP4Address& IP4Address::operator=(const sockaddr& addr)
			{
				if (addr.sa_family != AF_INET) throw FormatException("Unable to store address of non-IPv4 format in IPv4 container.");
				CopyMemory(&m_sockaddr, &addr, sizeof(m_sockaddr));
				return *this;
			}

			inline IP4Address& IP4Address::operator=(const IP4Address& cp)
			{
				CopyMemory(&m_sockaddr, &cp.m_sockaddr, sizeof(m_sockaddr));
				return *this;
			}

			inline IP4Address& IP4Address::operator=(IP4Address&& cp) noexcept
			{
				CopyMemory(&m_sockaddr, &cp.m_sockaddr, sizeof(m_sockaddr));
				return *this;
			}

			inline bool IP4Address::operator==(const IP4Address& cp) const
			{
				return memcmp(&m_sockaddr, &(cp.m_sockaddr), AddrSize) == 0;
			}

			inline bool IP4Address::operator!=(const IP4Address& cp) const
			{				
				return memcmp(&m_sockaddr, &(cp.m_sockaddr), AddrSize) != 0;
			}

			inline UInt16 IP4Address::GetPort() const { 
				assert(m_sockaddr.sa_family == AF_INET); 
				return ntohs( ((sockaddr_in *)&m_sockaddr)->sin_port ); 
			}

			inline UInt32 IP4Address::GetIP() const { 
				assert(m_sockaddr.sa_family == AF_INET); 
				return ntohl( ((sockaddr_in *)&m_sockaddr)->sin_addr.s_addr ); 
			}

			inline void	IP4Address::GetIP(byte& b1, byte& b2, byte& b3, byte& b4) const
			{
				assert(m_sockaddr.sa_family == AF_INET);
				UInt32 hIP = ntohl(((sockaddr_in&)m_sockaddr).sin_addr.s_addr);
				b1 = (byte)(hIP >> 24);
				b2 = (byte)(hIP >> 16);
				b3 = (byte)(hIP >> 8);
				b4 = (byte)(hIP);
			}

			inline string IP4Address::ToString(bool ShowPort) const
			{
				if (m_sockaddr.sa_family != AF_INET) return string(S("[Unprintable AF]"));
				sockaddr_in inet = *(sockaddr_in *)&m_sockaddr;
				string str1, str2;	
				if (inet.sin_addr.s_addr == ANY)
					str1 = S("ANY");
				else
				{
					byte b1, b2, b3, b4;
					GetIP(b1, b2, b3, b4);
					str1 = std::to_string((unsigned)b1) + S(".")
						+ std::to_string((unsigned)b2) + S(".")
						+ std::to_string((unsigned)b3) + S(".")
						+ std::to_string((unsigned)b4);
				}
				if (ShowPort)
				{
					int nPort = ntohs(inet.sin_port);
					if (nPort == ANY) str2 = S(":ANY");
					else str2 = S(":") + std::to_string(nPort);
					return str1 + str2;
				}
				else return str1;
			}

			inline /*static*/ bool IP4Address::TryParse(const string& strIn, IP4Address& Value, UInt16 nDefaultPort /*= 0*/)
			{
				size_t iDot;
				string str = strIn;
				string sub;
				UInt8 b1, b2, b3, b4;
				UInt16 port;

				iDot = str.find('.');
				if (iDot == string::npos) return false;
				sub = str.substr(0,iDot);
				if (!UInt8_TryParse(sub.c_str(), NumberStyles::Integer, b1)) return false;				
				str = str.substr(iDot+1);

				iDot = str.find('.');
				if (iDot == string::npos) return false;
				sub = str.substr(0,iDot);
				if (!UInt8_TryParse(sub.c_str(), NumberStyles::Integer, b2)) return false;				
				str = str.substr(iDot+1);

				iDot = str.find('.');
				if (iDot == string::npos) return false;
				sub = str.substr(0,iDot);
				if (!UInt8_TryParse(sub.c_str(), NumberStyles::Integer, b3)) return false;				
				str = str.substr(iDot+1);

				iDot = str.find(':');				
				if (iDot == string::npos)
				{
					if (!UInt8_TryParse(str.c_str(), NumberStyles::Integer, b4)) return false;				
					Value.Set(b1, b2, b3, b4, nDefaultPort);
					return true;
				}
				else
				{
					sub = str.substr(0,iDot);
					if (!UInt8_TryParse(sub.c_str(), NumberStyles::Integer, b4)) return false;				
					str = str.substr(iDot+1);					
					if (!UInt16_TryParse(str.c_str(), NumberStyles::Integer, port)) return false;				
					if (!port) port = nDefaultPort;
					Value.Set(b1, b2, b3, b4, port);
					return true;
				}
			}

			inline /*static*/ IP4Address IP4Address::Parse(const string& strIn, UInt16 nDefaultPort /*= 0*/)
			{
				IP4Address ret;
				if (!TryParse(strIn, ret, nDefaultPort)) throw FormatException("Unable to parse IP address.");
				return ret;
			}

			#if 0
			inline /*static*/ UInt32 IP4Address::ToHost(byte b1, byte b2, byte b3, byte b4)
			{
				return MakeU32( MakeU16( b4, b3 ), MakeU16( b2, b1 ) );
			}

			inline /*static*/ UInt32 IP4Address::ToNetwork(byte b1, byte b2, byte b3, byte b4)
			{
				// Network byte order is big endian, so if we are little endian we swap and if we are big endian
				// we just assemble naturally...

				if (IsLittleEndian())
					return MakeU32( MakeU16( b1, b2 ), MakeU16( b3, b4 ) );
				else				
					return MakeU32( MakeU16( b4, b3 ), MakeU16( b2, b1 ) );
			}
			#endif
		}
	}
}

#endif	// __Socket_IP4Address_h__

//	End of IP4Address.h


