/////////
//  Sockets.h
//  Windows Sockets 2 Manager
//  Copyright (C) 1999, 2014 by Wiley Black
/////////

#ifndef __Sockets_h__
#define __Sockets_h__

/** Table of contents and references **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			class Provider;
		}
	}
};

/** Dependencies **/

#include "../wbFoundation.h"

#if !defined(_WINDOWS)
#include <sys/socket.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/ioctl.h>
typedef int SOCKET;
#define INVALID_SOCKET (-1)
#define SOCKET_ERROR (-1)
#define SD_SEND SHUT_WR
#define closesocket(x) close(x)
#else
typedef int socklen_t;
#endif

#include "SocketExceptions.h"
#include "IP4Address.h"

/** Content **/

namespace wb
{
	namespace net
	{
		namespace sockets
		{
			class Provider
			{
				#ifdef _WINDOWS
				static int m_nInitialized;
				static WSADATA m_wsaData;

				void Initialize()
				{
					if (m_nInitialized == 0)
					{
						if (WSAStartup(SOCKETS_VERSION, &m_wsaData)) throw SocketException(S("Sockets initialization failed."));
					}
					m_nInitialized ++;
				}

				void Shutdown()
				{
					if (m_nInitialized > 1) m_nInitialized--;
					else if (m_nInitialized == 1)
					{
						WSACleanup();
						m_nInitialized = 0;
					}
				}
				#else
				void Initialize() { }
				void Shutdown() { }
				#endif

			public:
				#ifdef _WINDOWS
				Provider() { m_nInitialized = 0; Initialize(); }
				~Provider() { Shutdown(); }
				#else
				Provider() { }
				~Provider() { }
				#endif
				
				/** Objects returned from the Create...Socket() calls should be delete'd when complete. **/

				TCPServerSocket* CreateTCPServerSocket(IP4Address LocalBinding = IP4Address::AnyAddress, int nConnectionQueueSize = 100);
				TCPSocket* CreateTCPSocket();
				UDPSocket* CreateUDPSocket();
			};
			
			#if defined(PrimaryModule) && defined(_WINDOWS)
			int Provider::m_nInitialized = 0;
			WSADATA Provider::m_wsaData;		
			#endif
		}
	}
}

/** Late Dependencies **/

#include "TCPSocket.h"
#include "UDPSocket.h"

#ifndef _WINDOWS
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#endif

/** Implementations **/

namespace wb { 
	namespace net { 
		namespace sockets {

			/** Inline Methods **/

			inline TCPServerSocket* Provider::CreateTCPServerSocket(IP4Address LocalBinding, int nConnectionQueueSize) {
				return new TCPServerSocket(LocalBinding, nConnectionQueueSize);
			}

			inline TCPSocket* Provider::CreateTCPSocket() { return new TCPSocket(); }

			inline UDPSocket* Provider::CreateUDPSocket() { return new UDPSocket(); }
		}
	}
}

#endif	// __Sockets_h__

//  End of Sockets.h

