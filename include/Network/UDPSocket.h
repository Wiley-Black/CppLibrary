/////////
//  UDPSocket.h
//  UDP/IP protocol using sockets interface.
//  Copyright (C) 1999-2002, 2014 by Wiley Black
/////////
//	Improvements/TODO:
//		See TCPSocket.h.  May apply here as well.
/////////

#ifndef __Sockets_h__
	#error Do not include this header directly.  Include Sockets.h instead.
#endif

#ifndef __UDPSocket_h__
#define __UDPSocket_h__

/** Table of contents **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			class UDPSocket;
		}
	}
};

/** Dependencies **/

#include "../wbFoundation.h"
#include "Sockets.h"
#include "SocketExceptions.h"
#include "IP4Address.h"
#include "../IO/MemoryStream.h"

#ifdef _WINDOWS
#include <Ws2tcpip.h>					// For IP_DONTFRAGMENT option
#endif

/** Content **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			class IPacketReceiver
			{
			public:
				typedef ::wb::io::MemoryStream Packet;
				virtual void OnReceive(Packet& packet, const IP4Address& From, UDPSocket& Socket) = 0;
			};

			class UDPSocket 
			{
			public:
				typedef ::wb::io::MemoryStream Packet;

			private:
				SOCKET					m_socket;

				/** recvfrom() discards data if the buffer provided is not large enough.  Therefore we will
					reuse a 64KB buffer. **/
				Packet					m_Packet;

			protected:	
				friend class Provider;
				UDPSocket();

			public:					
				~UDPSocket();

					// Call this method before sending to a broadcast address to enable broadcast messages.
				void EnableBroadcast();

				/// <summary>Call SetDoNotFragment(true) to instruct the socket to avoid fragmenting packets.  Call SetDoNotFragment(false)
				/// to permit fragmenting of packets.</summary>
				void SetDoNotFragment(bool Enable);

				void	SetRxBufferSize(UInt32 Size);
				UInt32	GetRxBufferSize();					// In Linux, will typically be 2x the set size to allow kernel overhead.

					/** 
					Connect()

					Local:			If specified, requests the socket use this local (source) address.
									If specified as ANY:[Port], the service provider will assign the IP Address.
									If specified as [IP.IP.IP.IP]:ANY, the service provider will assign a port number.
					**/
				void Connect(IP4Address Local = IP4Address::AnyAddress);
				
				void Process(IPacketReceiver* pReceiver = nullptr);

				bool CanTx();				// If true, it is safe to transmit more data.
				void Send(Packet& packet, IP4Address Destination);
				void Send(void *pPacket, int nPacketLength, IP4Address Destination);

				IP4Address GetLocalAddress();
			};

			/** Implementation **/

			inline UDPSocket::UDPSocket()
			{
				m_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
				if (m_socket == INVALID_SOCKET)
					throw UDPSocketException(S("Could not open UDP Socket for network communications!"));

					/** Determine maximum packet size and save in our QoS.  Ensure it is at least 64 bytes. **/
				/*
						**  The following code DOES work.  Unfortunately, Microsoft Windows does not necessarily live up
							to the maximum packet size, in particular when receiving packets.

					int nOptSize = sizeof(m_qos.m_nMaxPacketSize);
					if( getsockopt( m_socket, SOL_SOCKET, SO_MAX_MSG_SIZE, (char *)&m_qos.m_nMaxPacketSize, &nOptSize ) 
						|| m_qos.m_nMaxPacketSize < 64 )
					{
						Destroy();
						throw NetworkError( S("Error initializing UDP packet size information!") );				
					}
				*/

					/** Place the socket in non-blocking mode **/				
				#ifdef _WINDOWS
				u_long ll = 1;
				if (ioctlsocket(m_socket, FIONBIO, &ll) == SOCKET_ERROR)
				#else
				UInt32 flags;
				if ((flags = fcntl(m_socket, F_GETFL, 0)) < 0
				 || (fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) < 0))
				#endif				
					throw UDPSocketException(S("Could not enter socket non-blocking mode: "), GetLastSocketError());
			}

			inline UDPSocket::~UDPSocket()
			{
				if (m_socket != INVALID_SOCKET) { closesocket(m_socket); m_socket = INVALID_SOCKET; }
			}

			inline void UDPSocket::Connect(IP4Address Local)
			{
					/** Specify (to the socket) our local (source) address.  The address may be specific, ANY:ANY, 
						[IP.IP.IP.IP]:ANY, or ANY:[Port].  The definition of ANY in IP4Address is such that
						bind() will recognize it. **/
				if (::bind(m_socket, &(Local.m_sockaddr), sizeof(sockaddr_in)) == SOCKET_ERROR)									
					throw UDPSocketException(S("Error binding to UDP socket: "), GetLastSocketError());				
			}
			
			inline IP4Address UDPSocket::GetLocalAddress()
			{
				sockaddr saDest; socklen_t saLen = sizeof(saDest);
				if (getsockname(m_socket, &saDest, &saLen) == SOCKET_ERROR) return IP4Address::AnyAddress;
				if (saDest.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
				return IP4Address(saDest);
			}

			inline void UDPSocket::EnableBroadcast()
			{
				UInt32	bYes = true;
				if (setsockopt( m_socket, SOL_SOCKET, SO_BROADCAST, (const char *)&bYes, sizeof(bYes) ) == SOCKET_ERROR)
					throw UDPSocketException(S("Error enabling broadcast on socket: "), GetLastSocketError());
			}

			inline void UDPSocket::SetDoNotFragment(bool Enable)
			{
				#ifdef _WINDOWS
				UInt32	bYes = Enable;
				if (setsockopt(m_socket, IPPROTO_IP, IP_DONTFRAGMENT, (const char *)&bYes, sizeof(bYes)) == SOCKET_ERROR)
					throw UDPSocketException(S("Error setting do-not-fragment on socket: "), GetLastSocketError());				
				#elif defined(_LINUX)
				UInt32	Value = Enable ? IP_PMTUDISC_DO : IP_PMTUDISC_DONT;
				if (setsockopt(m_socket, IPPROTO_IP, IP_MTU_DISCOVER, (const char *)&Value, sizeof(Value)) == SOCKET_ERROR)
					throw UDPSocketException(S("Error setting do-not-fragment on socket: "), GetLastSocketError());
				#else
				UInt32	bYes = Enable;
				if (setsockopt(m_socket, IPPROTO_IP, IP_DONTFRAG, (const char *)&bYes, sizeof(bYes)) == SOCKET_ERROR)
					throw UDPSocketException(S("Error setting do-not-fragment on socket: "), GetLastSocketError());
				#endif
			}

			inline void UDPSocket::SetRxBufferSize(UInt32 Size)
			{				
				if (setsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char *)&Size, sizeof(Size)) == SOCKET_ERROR)
					throw UDPSocketException("Unable to configure socket receive buffer size: ", GetLastSocketError());
			}

			inline UInt32 UDPSocket::GetRxBufferSize()
			{
				UInt32 ret;
				socklen_t param_size = sizeof(ret);
				if (getsockopt(m_socket, SOL_SOCKET, SO_RCVBUF, (char *)&ret, &param_size) == SOCKET_ERROR)
					throw UDPSocketException("Unable to read socket receive buffer size: ", GetLastSocketError());
				return ret;
			}
			
			inline void UDPSocket::Send(Packet& packet, IP4Address Destination)
			{
				packet.Rewind();
				if (sendto(m_socket, (char *)packet.GetDirectAccess(), (UInt32)MinOf((Int64)UInt32_MaxValue, packet.GetLength()),
							0, (struct sockaddr *)&(Destination.m_sockaddr), sizeof(sockaddr) ) == SOCKET_ERROR)
					throw UDPSocketException(S("Error transmitting UDP packet: "), 
						GetLastSocketError(), UDPSocketException::g_SendErrors);
			}

			inline void UDPSocket::Send(void *pPacket, int nPacketLength, IP4Address Destination)
			{
				if (sendto(m_socket, (char *)pPacket, nPacketLength,
							0, (struct sockaddr *)&(Destination.m_sockaddr), sizeof(sockaddr) ) == SOCKET_ERROR)
					throw UDPSocketException(S("Error transmitting UDP packet: "), 
						GetLastSocketError(), UDPSocketException::g_SendErrors);
			}			

			inline bool UDPSocket::CanTx()
			{
				// The FD_SET macro can generate a warning about expression has no effect because it wraps
				// in a do { } while(0,0) clause.  In the nvcc compiler, I don't see how to disable that
				// warning.  As a workaround, the WinSock2.h definition from 3-14-2021 is copied here but
				// the while(0,0) is removed.
				#define WORKAROUND_FD_SET(fd, set) { \
					u_int __i; \
					for (__i = 0; __i < ((fd_set FAR *)(set))->fd_count; __i++) { \
						if (((fd_set FAR *)(set))->fd_array[__i] == (fd)) { \
							break; \
						} \
					} \
					if (__i == ((fd_set FAR *)(set))->fd_count) { \
						if (((fd_set FAR *)(set))->fd_count < FD_SETSIZE) { \
							((fd_set FAR *)(set))->fd_array[__i] = (fd); \
							((fd_set FAR *)(set))->fd_count++; \
						} \
					} \
				}				

				fd_set	fd_write;
				FD_ZERO( &fd_write );				
				WORKAROUND_FD_SET( m_socket, &fd_write );

				#undef WORKAROUND_FD_SET

				timeval	tval;
				tval.tv_sec = tval.tv_usec = 0;

				if( select( 0, NULL, &fd_write, NULL, &tval ) == SOCKET_ERROR )
					throw UDPSocketException(S("Error checking outbound queue on UDP Socket: "), GetLastSocketError());

				return FD_ISSET( m_socket, &fd_write ) ? true : false;
			}

			inline void UDPSocket::Process(IPacketReceiver* pReceiver)
			{				
				// Get ready to receive...					
				IP4Address FromAddr;

				#if 0
				/** Method 1: use ioctlsocket(FIONREAD) to predict the buffer size required.  This appears to work on MS Windows. **/

					/** Use ioctlsocket( FIONREAD ) to determine amount of data pending in the network's input buffer.
						For datagrams, this returns the size of the first datagram queued on the socket.  
						Note: Zero-length datagrams are possible. **/
				UInt32 lSize = 0;
				if( ioctlsocket( m_socket, FIONREAD, &lSize ) == SOCKET_ERROR )
					throw UDPSocketException(S("Error querying UDP Socket incoming length: "), 
						GetLastSocketError(), UDPSocketException::g_IOCtlErrors);					
					
				m_Packet.EnsureCapacity(lSize);				
				m_Packet.Rewind();

					/** Now that we have allocated a buffer (packet) for the queued datagram, retrieve the datagram
						from the socket. **/
				sockaddr saFrom; socklen_t saLen = sizeof(saFrom);
				int nValue = recvfrom(m_socket, (char *)m_Packet.GetDirectAccess(), m_Packet.GetCapacity(), 0, 
										&saFrom, (socklen_t *)&saLen );
				if (nValue == SOCKET_ERROR)
				{
					if (WouldBlock(GetLastSocketError())) return;
					throw UDPSocketException(S("Error retrieving UDP packet: "), 
						GetLastSocketError(), UDPSocketException::g_RecvErrors);
				}
				if (saFrom.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
				FromAddr = saFrom;

					/** Send the datagram up. **/

				m_Packet.SetLength(nValue);
				m_Packet.Rewind();
				if (pReceiver != nullptr) pReceiver->OnReceive(m_Packet, FromAddr, *this);				

				#else					
				/** Method 2: Only use recvfrom(), and always allocate the maximum possible packet size.  This method appears to work under MS Windows. **/

					/** Attempt to retrieve any waiting datagram from the socket.  Unfortunately, recvfrom()
						will discard data if we supply too-small of a buffer, so to be certain we need
						a buffer that covers the maximum packet size.  For UDP, this is the MTU, but the
						absolute maximum packet size is 64k.  We will therefore use that and reuse the
						buffer. **/
				m_Packet.EnsureCapacity(65535);
				sockaddr saFrom; socklen_t saLen = sizeof(saFrom);
				int nValue = recvfrom(m_socket, (char *)m_Packet.GetDirectAccess(0), (UInt32)MinOf((Int64)UInt32_MaxValue, m_Packet.GetCapacity()), 
						0, &saFrom, &saLen );
				if (nValue == SOCKET_ERROR)
				{
					int serr = GetLastSocketError();
					if (WouldBlock(serr)) return;
					throw UDPSocketException(S("Error retrieving UDP packet: "), serr, UDPSocketException::g_RecvErrors);
				}
				// printf("UDP received %i byte packet.\n", nValue);
				if (saFrom.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
				FromAddr = saFrom;

					/** Send the datagram up. **/

				m_Packet.SetLength(nValue);
				m_Packet.Rewind();
				if (pReceiver != nullptr) pReceiver->OnReceive(m_Packet, FromAddr, *this);

				#endif

			}// End Process()
		}
	}
}

#endif	// __UDPSocket_h__

//	End of UDPSocket.h



