/////////
//  TCPSocket.h
//  TCP/IP protocol using sockets interface.
//  Copyright (C) 1999-2002, 2014 by Wiley Black
/////////
/*
	Notes:
		Provides an asynchronous (non-blocking) socket implementation.

	Sample usage for clients:			

		------------------------------------------------------------------------------------------------------------
		using namespace wb::net;
		sockets::Provider	ServiceProvider;		// Usually an application global.

		sockets::TCPSocket*	pMySocket = ServiceProvider.CreateTCPSocket();
  
		IP4Address	addrSource( IP4Address::ANY, 99 );		// IP Address of "ANY",   Port Number of 99.
		IP4Address	addrDestination( 1, 2, 3, 4, 99 );		// IP Address of 1.2.3.4, Port Number of 99.
			
		pMySocket->BeginConnect(addrDestination, addrSource);
		while (!pMySocket->IsConnected()) pMySocket->Process();

		const TCHAR *lpszTransmission	= "Hello World!";
		while (!pMySocket->IsTxReady()) pMySocket->Process();
		pMySocket->Add(lpszTransmission, _tcslen(lpszTransmission));

		pMySocket->BeginDisconnect();
		while (pMySocket->IsConnected()) 
		{
			pMySocket->Process();					// Waits for graceful closure by remote as well.  A timeout advised.

			// The connection can only close gracefully after all data has been read from the remote, even if discarded.
			if (pMySocket->IsRxReady()) {
				byte trash;
				pMySocket->Get(trash);				// Discard received data.
			}
		}		

		delete pMySocket;
		------------------------------------------------------------------------------------------------------------

	Sample usage for servers:	

		------------------------------------------------------------------------------------------------------------
		using namespace wb::net;
		sockets::Provider	ServiceProvider;		// Usually an application global.

			// IP Address of "ANY" and port number of 99...
		IP4Address BindAddress(IP4Address::ANY, 99);
		sockets::TCPServerSocket	*pMyServer = ServiceProvider.CreateTCPServerSocket(BindAddress);

		while( WANT_TO_BE_SERVER )
		{
			TCPSocket*		pMySocket	= pMyServer->AcceptConnection();
				// AcceptConnection(), for servers, returns NULL if no incoming connections are pending...
			if (pMySocket == nullptr) continue;

				// Server sockets are already connected at this point.

				// In this example, we demonstrate only the handling of 1 server socket at a time.  In most real-world
				// applications, we would instead want to add to a list (possibly an array) of sockets process all of 
				// them routinely, deleting them when they lose their connection and removing them from the list/array.

			while (pMySocket->IsConnected())
			{
				pMySocket->Process();

				if (pMySocket->GetRxCount())
				{
					byte ch;
					pMySocket->Get(ch);
					_putch(ch);
				}
				
				if (NOTHING_MORE_TO_SEND) pMySocket->BeginDisconnect();		// Usually want a timeout on disconnection/inactivity.
			}

			delete pMySocket;

		}// End while( WANT_TO_BE_SERVER )

		delete pMyServer;
		------------------------------------------------------------------------------------------------------------
*/
/////////

#ifndef __Sockets_h__
	#error Do not include this header directly.  Include Sockets.h instead.
#endif

#ifndef __TCPSocket_h__
#define __TCPSocket_h__

/** Table of contents **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			class TCPSocket;
			class TCPServerSocket;
		}
	}
};

/** Dependencies **/

#include "wbFoundation.h"
#include "Sockets.h"
#include "SocketExceptions.h"
#include "IP4Address.h"
#include "../Memory Management/FixedRingBuffer.h"

/** Content **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			#ifdef _WINDOWS
			inline int GetLastSocketError() { return WSAGetLastError(); }
			inline bool WouldBlock(int ErrorCode) { return ErrorCode == WSAEWOULDBLOCK; }
			#else
			inline int GetLastSocketError() { return errno; }
			inline bool WouldBlock(int ErrorCode) { return ErrorCode == EAGAIN || ErrorCode == EWOULDBLOCK || ErrorCode == EINPROGRESS; }
			#endif

			class TCPServerSocket
			{
				SOCKET m_socket;				

			public:
				/** 
					Local:			If specified, requests the socket will serve on the specified local address.
									If null, the service provider chooses the address.
									If specified as ANY:[Port], the service provider will assign only the IP Address (recommended).
									If specified as [IP.IP.IP.IP]:ANY, the service provider will assign only a port number.
				**/
				TCPServerSocket(const IP4Address& Local, int nConnectionQueueSize = 100);

				~TCPServerSocket();				

				IP4Address GetLocalAddress();
				TCPSocket* AcceptConnection();
			};	

			class TCPSocket
			{
				SOCKET m_socket;

			public:
				IP4Address		m_addrLocalAddress;
				IP4Address		m_addrDestinationAddress;

				#ifdef _WINDOWS
				HANDLE			m_heNetwork;					// Network event indicator	
				#endif

					// The TX Buffer is not normally used, however in the event that an Add() call which suprisingly comes back
					// indicating "WSAEWOULDBLOCK" we will buffer just one thing.  We have to be careful to send this out before
					// closing the link too.
				byte			*m_pTxBuffer;
				uint			m_nTxPending;					// Number of bytes in m_pTxBuffer which are pending.
				uint			m_nTxAlloc;						// Number of bytes alloctated in m_pTxBuffer.				

					// We use an RX buffer only because sockets make it impossible to distinguish when an EOF event has
					// occurred without actually reading data.  If data is requested larger than the buffer, then a
					// call to the underlying system is made directly.
				enum { RxBufferSize = 1024 };
				wb::memory::FixedRingBuffer		m_RxBuffer;

				enum_class_start(States,int)
				{
					Disconnected,
					Connecting,
					Connected
				}
				enum_class_end(States);

				States			m_State;
				bool			m_bCanWrite;					// Indicates writing on the port is currently allowed w/o blocking.  (Last we heard.)				
				bool			m_bSendPending;					// Indicates that we want to flush the transmit buffer as soon as the queue is empty (Windows only)
				bool			m_bWriteClosePending;			// Indicates that we want to close the write pipe as soon as the queue is empty	

				bool			m_bLocalFinished;				// Indicates the transmit side (at least) has been closed.
				bool			m_bRemoteFinished;				// Indicates the remote is finishing up

				void	OnSendError( int nErrorCode );
				void	OnRecvError( int nErrorCode );

				#if 0
				class QoS : public QualityOfService 
				{
					friend class TCPSocket;

					dword	m_dwTxBandwidth;
					dword	m_dwMaxConnectionBacklog;

				public:
					virtual bool IsIntegrity() const { return true; }
					virtual bool IsGuaranteed() const { return true; }
					virtual bool IsSequenced() const { return true; }
					virtual bool IsFlowControl() const { return false; }
					virtual UInt32 GetMaxPacketSize() const { return UInt32_MaxValue; }

						/** 
							MaxConnectionBacklog applies only to server (listening) sockets.  It specifies the
							number of connections which may be "backlogged".  That is, it allows some number of clients
							who are trying to connect with our server to be backlogged until we can accept their
							incoming connection with the Connect() function.
						**/
					virtual UInt32 GetMaxConnectionBacklog() const { return m_dwMaxConnectionBacklog; }
					virtual UInt32 GetBandwidth() const { return m_dwTxBandwidth; }

					QoS(UInt32 dwTxBandwidth = 2000, UInt32 dwMaxConnectionBacklog = SOMAXCONN){ m_dwTxBandwidth = dwTxBandwidth; m_dwMaxConnectionBacklog = dwMaxConnectionBacklog; }
					QoS(const CQoS& cp){ m_dwTxBandwidth = cp.GetBandwidth(); m_dwMaxConnectionBacklog = cp.GetMaxConnectionBacklog(); }

				} m_qos;
				#endif

			protected:	
				friend class Provider;
				friend class TCPServerSocket;

				TCPSocket();
				TCPSocket(SOCKET s, const IP4Address& Destination);

			public:
	
				~TCPSocket();

					/** Call EnableBroadcast() prior to BeginConnect() to enable the use of broadcast addresses on this socket. **/
				void EnableBroadcast();

					/** 
					BeginConnect()

					For TCPSocket objects created via TCPServerSocket (Servers), calling Connect() is optional.  It may be
					used to specify the quality-of-service, or may have no effect (but will return success.)

					Destination:	For TCPSocket objects created via TCPServerSocket (Servers), this parameter is ignored.
									If specified, actively establishes a connection with this destination.
									If specified as [IP.IP.IP.IP]:ANY, the service provider will assign a port number.
									If null for a client, an error occurs.
							NOTE:	This parameter should not be changed between calls to Connect(), unless BeginDisconnect() is called.

					Local:			For TCPSocket objects created via TCPServerSocket (Servers), this parameter is ignored.
									If specified, requests the socket use this local (source) address.
									If null, the service provider chooses the address.
									If specified as ANY:[Port], the service provider will assign the IP Address (recommended).
									If specified as [IP.IP.IP.IP]:ANY, the service provider will assign a port number.
							NOTE:	This parameter should not be changed between calls to Connect(), unless BeginDisconnect() is called.
							NOTE:	Microsoft recommends that clients use a port number of ANY to reduce conflicts.

					pQOS:			For most applications, should be specified, providing a bandwidth setting from user.
									If the application will use small bandwidth (less than 2KB/sec), the default may be used.
									If default, then the CQualityOfService settings are not changed.
									If this parameter changes between calls to Connect(), the value from the last call is used.
					**/
				virtual void BeginConnect(
					IP4Address Destination = IP4Address::AnyAddress, 
					IP4Address Local = IP4Address::AnyAddress 
					//QualityOfService *pQOS = g_pQoSNone 
					);

					/** IsConnecting() and IsConnected()

						The IsConnected() method indicates if a connection is
						currently established.  In TCP, further details on the connection may be
						retrieved from IsLocalFinished() and IsRemoteFinished().

						The IsConnecting() method indicates that a connection has been initiated
						by calling Connect(), but has not yet reached the connected state, nor has
						it yet reached a failure state.
					**/
				virtual bool IsConnected() { return (m_State == States::Connected); }
				virtual bool IsConnecting() { return (m_State == States::Connecting); }

					/** IsRemoteFinished()
						Returns true if the receive side of the connection has completed its transmission.  This happens
						when the remote has called BeginDisconnect() and all data prior to the BeginDisconnect() has been 
						received (via Get calls.)  If no longer connected, IsRemoteFinished() also returns true.
					**/
				virtual bool IsRemoteFinished();

					/** IsLocalFinished()
						Returns true if the send side of the connection has completed closure.  This happens after a
						BeginDisconnect() call and all outgoing data has been transmitted.  If no longer connected,
						IsLocalFinished() also returns true.
					**/
				virtual bool IsLocalFinished();

					/** BeginDisconnect()
			
						This method will begin closure of the transmit side of the connection.  The receive side will still
						be enabled until IsRemoteFinished() returns true or a hard/abortive close occurs (by deleting the
						TCPSocket object).  

						There are four methods for closing a connection:

							1. Hard/Abortive close.  For this, delete the TCPSocket object.

							2. Close by Remote.  In this situation, the local side will close as soon as the remote
								has completed transmission.  To implement:
								a. Operate until IsRemoteFinished() returns true.
								b. Call BeginDisconnect().
								c. Call Process() until IsConnected() returns false.

							3. Close at completion.  In this situation, both sides will continue listening until the
								other side has completed its transmission.  To implement:
								a. Operate until ready to close the transmission link.  Watch for !IsConnected().
								b. Call BeginDisconnect().
								c. Continue processing received data until IsRemoteFinished() returns true.					
								e. Continue calling Process() until IsConnected() returns false.
								f. Delete the TCPSocket object.

							4. Close by Local.  In the same way as case 1, we will be disregarding any 
								received data after interest has been lost.  However, we will ensure that all data 
								we are transmitting is received by the remote.  For this case:
								a. Operate until ready to close the transmission link.  Watch for !IsConnected().
								b. Call BeginDisconnect() (and Process()) until IsLocalFinished() returns true.
								c. Delete the TCPSocket object.

						Calling BeginDisconnect() will begin a graceful close of the connection.  Poll the IsConnected() method
						until false is returned for disconnect completion.  To issue an hard/abortive close destroy the 
						TCPSocket object.  
			
						If received data must be received entirely, then wait for IsRemoteFinished() to return true before
						deleting the TCPSocket object.

						Polling IsConnected() for false is important to ensure transmitted data completion.  The TCPSocket 
						may have queued transmission data and may not have been able to transmit the queued data and/or the 
						FIN (Finish) even if Flush() and BeginDisconnect() calls have been made.  For the queued data to be 
						completely transferred, the caller must continue calling Process() until IsConnected() returns false.

						The caller must also retrieve all incoming data in order that IsConnected() can transition to false.
						The received data can be discarded, but the Get() calls must be made.  This is because the driver
						may not indicate the FIN condition from the remote until it reaches it in the sequence.
					**/
				virtual void BeginDisconnect();				

					/* Communications */

				virtual void Process();

				bool IsTxReady() { return !IsTxFull(); }
				virtual bool IsTxFull(){ Process(); return IsLocalFinished() || !m_bCanWrite; }
				virtual UInt32 GetTxCount(){ return 0; }
				virtual UInt32 GetTxFree(){ return IsTxFull() ? 0 : 1500; }	

				virtual void Add(byte b);
				virtual void Add(void *pBlock, Int32 nLength);
				virtual void Add(const char *lpsz);
				virtual void Add(const string&);

				virtual void Flush();

				virtual void Get(byte& b);
				virtual void Get(void *pBlock, UInt32 dwLength);
 
				virtual bool IsRxReady();
				virtual uint GetRxCount();

				// virtual CQualityOfService *GetQoS(){ return &m_qos; }

					/** TCP Specific **/

				const IP4Address& GetLocalAddress() const { return m_addrLocalAddress; }
				const IP4Address& GetDestinationAddress() const { return m_addrDestinationAddress; }

					/** Queing Systems (May include estimations) **/		
					/**
						CStreamCommunication classes will typically want to return their own queue sizes.
						Others will usually want to return the lower layer's queue sizes.
					**/
				virtual uint GetRxQueueSize(){ return 2048000; }	// Actual information not available with WinSockets.
				virtual uint GetTxQueueSize(){ return 1024000; }	// Actual information not available with WinSockets.
					// Actual information not available with WinSockets.  Overhead based on typical UDP/IP implementation.
				virtual uint GetOverhead(uint dwPerBytes){ return 8 + 24 + (dwPerBytes / 256); }
			};

			/** Implementation, TCPServerSocket **/

			inline TCPServerSocket::TCPServerSocket(const IP4Address& Local, int nConnectionQueueSize)
			{
					/** Open the server socket **/
				m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
				if (m_socket == INVALID_SOCKET)
					throw SocketException(S("Could not open TCP Server Socket for network communications!"));

					/** Decide and validate what our local address will be **/				
		
				#ifdef _DEBUG
					/** When don't we want this?  It only prevents re-use of the same PORT on a single host.
						This usually happens when you run your server program and some test program to localhost,
						using the same port.  This isn't going to work, so by turning off SO_REUSEADDR we at
						least get an error message.
					**/

					/** When do we want this?  Sometimes a program crash (i.e. caused by Norton Internet Security
						when debugging) causes the port to be latched onto.  Enabling this frees us from having
						to reboot during debugging.
					**/

					/** Enable/Disable re-use of the socket address **/
				UInt32 bYes = true;
				if( setsockopt( m_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&bYes, sizeof(bYes) ) == SOCKET_ERROR )				
					throw TCPSocketException(S("Error enabling re-use of Socket address: "), GetLastSocketError(), TCPSocketException::g_SetOptErrors);				
				#endif

					/** Specify (to the socket) our local (source) address.  The address may be specific, ANY:ANY, 
						[IP.IP.IP.IP]:ANY, or ANY:[Port].  The definition of ANY in IP4Address is such that
						bind() will recognize it. **/
				if (::bind( m_socket, &(Local.m_sockaddr), Local.AddrSize ) == SOCKET_ERROR)
					throw TCPSocketException(S("Error binding TCP Server Socket to address ") + Local.ToString(true) + S(": "),
						GetLastSocketError(), TCPSocketException::g_BindErrors);

					// Place the socket into a state where it is listening for new connections.  The individual connections
					// are retrieved by calling accept().
				if (listen( m_socket, nConnectionQueueSize ) == SOCKET_ERROR)
					throw TCPSocketException(S("Could not establish TCP Server Socket: "), GetLastSocketError());
	
					/** Place the server socket in non-blocking mode **/				
				#ifdef _WINDOWS
				u_long ll = 1;
				if (ioctlsocket( m_socket, FIONBIO, &ll ) == SOCKET_ERROR)
				#else
				UInt32 flags;
				if ((flags = fcntl(m_socket, F_GETFL, 0)) < 0
				 || (fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) < 0))
				#endif
					throw TCPSocketException(S("Could not enter TCP Server socket non-blocking mode: "), GetLastSocketError());
			}

			inline TCPServerSocket::~TCPServerSocket()
			{
				if( m_socket != INVALID_SOCKET )
				{
					closesocket(m_socket);
					m_socket = INVALID_SOCKET;
				}
			}

			inline TCPSocket* TCPServerSocket::AcceptConnection()
			{				
				if (m_socket == INVALID_SOCKET) throw TCPSocketException(S("Invalid server socket."));

					/** An incoming connection may be waiting for us...we can call accept() and expect non-blocking response. **/				
				sockaddr saDest; socklen_t saLen = sizeof(saDest);
				SOCKET s = accept( m_socket, &saDest, &saLen);
				if (s == INVALID_SOCKET)
				{
					int nErrorCode = GetLastSocketError();
					if (WouldBlock(nErrorCode)) return NULL;		// This error simply indicates no pending connection requests.

					throw TCPSocketException(S("Error accepting incoming TCP connection on socket: "), 
						nErrorCode, TCPSocketException::g_AcceptErrors);
				}
				if (saDest.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");

				return new TCPSocket(s, IP4Address(saDest));
			}

			inline IP4Address TCPServerSocket::GetLocalAddress()
			{				
				sockaddr saDest; socklen_t saLen = sizeof(saDest);
				if (getsockname(m_socket, &saDest, &saLen) == SOCKET_ERROR) return IP4Address::AnyAddress;
				if (saDest.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
				return IP4Address(saDest);
			}

			/** Implementation, TCPSocket **/

			inline TCPSocket::TCPSocket() : m_RxBuffer(RxBufferSize)
			{
				m_pTxBuffer		= nullptr;
				m_nTxPending	= 0;
				m_nTxAlloc		= 0;

				m_State					= States::Disconnected;
				m_bCanWrite				= false;
				m_bSendPending			= false;
				m_bWriteClosePending	= false;
	
				m_bLocalFinished		= false;
				m_bRemoteFinished		= false;

				m_socket		= INVALID_SOCKET;				
				#ifdef _WINDOWS
				m_heNetwork		= WSA_INVALID_EVENT;
				#endif

				if (m_socket == INVALID_SOCKET)
				{
					m_socket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
					if (m_socket == INVALID_SOCKET)					
						throw TCPSocketException(S("Could not allocate TCP Socket for network communications!"));
				}
			}

			inline TCPSocket::TCPSocket(SOCKET s, const IP4Address& Destination) : m_RxBuffer(RxBufferSize)
			{								
				m_pTxBuffer		= NULL;
				m_nTxPending	= 0;
				m_nTxAlloc		= 0;	

					// Server-created sockets are already connected at construction of the TCPSocket object.  See TCPServerSocket.
				m_State					= States::Connected;
				m_bCanWrite				= false;
				m_bSendPending			= false;
				m_bWriteClosePending	= false;
	
				m_bLocalFinished		= false;
				m_bRemoteFinished		= false;

				#ifdef _WINDOWS
				m_heNetwork				= WSA_INVALID_EVENT;
				#endif

				m_socket				= s;
				m_addrDestinationAddress= Destination;

					// Although already connected, we need to configure the socket...
				BeginConnect(Destination, m_addrLocalAddress);					
			}

			inline void TCPSocket::EnableBroadcast()
			{
				int enable = 1;
				if (setsockopt (m_socket, SOL_SOCKET, SO_BROADCAST, (char*)&enable, sizeof (enable)) == SOCKET_ERROR)
					throw TCPSocketException(S("Unable to enable broadcast on socket: "), GetLastSocketError());
			}

			inline void TCPSocket::BeginConnect(IP4Address Destination, IP4Address Local)
						//CQualityOfService *pQoS /*= g_pQoSNone*/ )
			{
				/** Issues to consider regarding multiple calls to Connect():

					Blocking or Non-Blocking?			This implementation is non-blocking.
				**/
			
				assert (m_socket != INVALID_SOCKET);

				#ifdef _WINDOWS
				if (m_heNetwork == WSA_INVALID_EVENT) 
				{
						/** Setup Events Handle **/

					m_heNetwork = WSACreateEvent();
					if (m_heNetwork == WSA_INVALID_EVENT)
						throw TCPSocketException(S("Could not allocate required socket-event resource: "), GetLastSocketError());
				}
				#endif

					/** Setup Quality-of-Service **/

				//if (pQoS && pQoS->IsKindOf( EMBEDDED_DYNAMIC_INFO( CTCPSocket, CQoS ) ) ) m_qos = *(CQoS *)pQoS;					

					/** Place the socket in non-blocking mode **/
				
				#ifdef _WINDOWS
				u_long ll = 1;
				if (ioctlsocket( m_socket, FIONBIO, &ll ) == SOCKET_ERROR)
				#else
				UInt32 flags;
				if ((flags = fcntl(m_socket, F_GETFL, 0)) < 0
				 || (fcntl(m_socket, F_SETFL, flags | O_NONBLOCK) < 0))
				#endif
				{
					throw TCPSocketException(S("Could not enter socket non-blocking mode: "), GetLastSocketError());
				}

				if (m_State == States::Disconnected)
				{
						/**** Client Sockets ****/

					assert (m_socket != INVALID_SOCKET);
					assert (Destination.GetIP() != IP4Address::ANY);	// Client socket Connect() calls require an IP Address specification.				

					#ifdef _DEBUG
						/** Why don't we want this?  It only prevents re-use of the same PORT on a single host.
							This usually happens when you run your server program and some test program to localhost,
							using the same port.  This isn't going to work, so by turning off SO_REUSEADDR we at
							least get an error message.
						**/

						/** Why do we want this?  Sometimes a program crash (often caused by Norton Internet Security
							when debugging) causes the port to be latched onto.  Enabling this frees us from having
							to reboot during debugging.
						**/

						/** Enable re-use (for client sockets only) of the socket address **/
					UInt32 bYes = 1;
					if (setsockopt( m_socket, SOL_SOCKET, SO_REUSEADDR, (const char *)&bYes, sizeof(bYes) ) == SOCKET_ERROR)
					{
						throw TCPSocketException(S("Error enabling re-use of Socket address: "), 
							GetLastSocketError(), TCPSocketException::g_SetOptErrors);					
					}
					#endif

						/** Localhost connections:

							In order to do a localhost connection, the destination and source port #'s have to be
							different!
						**/

						/** Specify (to the socket) our local (source) address.  The address may be specific, ANY:ANY, 
							[IP.IP.IP.IP]:ANY, or ANY:[Port].  The definition of ANY in IP4Address is such that
							bind() will recognize it. **/
					sockaddr saDest;
					if (::bind(m_socket, &saDest, sizeof(saDest)) == SOCKET_ERROR)
						throw TCPSocketException(S("Error binding TCP Socket to address ") + Local.ToString(true) + S(": "),
							GetLastSocketError(), TCPSocketException::g_BindErrors);
					if (saDest.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
					Local = saDest;
				}

					/** Attempt to determine the local address bound to for later queries **/							
			
				sockaddr saDest; socklen_t saLen = sizeof(saDest);
				if (getsockname(m_socket, &saDest, &saLen) == SOCKET_ERROR)
					m_addrLocalAddress = IP4Address::AnyAddress;									
				else
				{
					if (saDest.sa_family != AF_INET) throw NotSupportedException("Only IPv4 addresses are supported.");
					m_addrLocalAddress = saDest;
				}

					/** Now that we've completed bind(), setup for notification on certain events **/
		
				#ifdef _WINDOWS
				if (WSAEventSelect( m_socket, m_heNetwork, FD_WRITE | FD_CONNECT ) == SOCKET_ERROR)
					throw TCPSocketException(S("Could not enable event selection on socket: "), GetLastSocketError());				
				#endif
			
				if (m_State == States::Disconnected)
				{
						/** Client socket - initiate the connection **/

						/** If connecting to localhost, check notes at 'Bind()' above. **/

					if (connect(m_socket, &(Destination.m_sockaddr), sizeof(sockaddr_in)) != SOCKET_ERROR)
					{
						m_State = States::Connected;
					}
					else
					{
						if (WouldBlock(GetLastSocketError()))
							m_State = States::Connecting;
						else
							throw TCPSocketException(S("Error connecting to remote host ") + Destination.ToString(true) + S(": "), 
								GetLastSocketError(), TCPSocketException::g_ConnectionErrors);
					}

					m_addrDestinationAddress = Destination;
				}
				
				// If not already connected, the process continues asynchronously.  See Process().
			}

			inline void TCPSocket::BeginDisconnect()
			{
				if (!m_nTxPending){
					if (!m_bLocalFinished) shutdown(m_socket, SD_SEND);
					m_bLocalFinished = true;
				}
				else
				{
					m_bWriteClosePending = true; 					
				}
			}

			inline TCPSocket::~TCPSocket()
			{
				if (m_socket != INVALID_SOCKET)
				{
					closesocket( m_socket );
					m_socket = INVALID_SOCKET;
				}

				#ifdef _WINDOWS
				if (m_heNetwork != WSA_INVALID_EVENT)
				{
					WSACloseEvent( m_heNetwork );
					m_heNetwork = WSA_INVALID_EVENT;
				}
				#endif

				if (m_pTxBuffer)
				{
					delete[] m_pTxBuffer;
					m_pTxBuffer			= nullptr;
					m_nTxAlloc			= 0;
				}				

				m_nTxPending			= 0;

				m_State = States::Disconnected;				
				m_bCanWrite				= false;
				m_bSendPending			= false;
				m_bWriteClosePending	= false;
	
				m_bLocalFinished		= false;
				m_bRemoteFinished		= false;
			}

			inline void TCPSocket::Process()
			{
				if (m_socket == INVALID_SOCKET) return;

				#ifdef _WINDOWS
				WSANETWORKEVENTS	NetworkEvents;
				if( WSAEnumNetworkEvents( m_socket, m_heNetwork, &NetworkEvents ) == SOCKET_ERROR )				
					throw TCPSocketException(GetLastSocketError());

				if (NetworkEvents.lNetworkEvents & FD_CONNECT)
				{
					if( NetworkEvents.iErrorCode[FD_CONNECT_BIT] != 0 )
					{
						m_State = States::Disconnected;
						throw TCPSocketException(S("Error while attempting to connect: "), NetworkEvents.iErrorCode[FD_CONNECT_BIT]);
					}
					else
					{
						m_State = States::Connected;						
						m_bCanWrite = true;
					}
				}				

				if (!m_bLocalFinished && IsConnected())
				{
					if( NetworkEvents.lNetworkEvents & FD_WRITE )
					{
						if( NetworkEvents.iErrorCode[FD_WRITE_BIT] != 0 )						
							throw TCPSocketException(S("Error while attempting to transmit: "), NetworkEvents.iErrorCode[FD_WRITE_BIT]);						
						else
						{
							if( m_nTxPending )
							{
								assert( m_pTxBuffer );

								if( send( m_socket, (const char *)m_pTxBuffer, m_nTxPending, 0 ) == SOCKET_ERROR ){
									int status = GetLastSocketError();
									if (WouldBlock(status)) m_bCanWrite = false;
									else OnSendError(status);
								} else {
									m_nTxPending = 0;
									m_bCanWrite = true;				
								}
							}
							else m_bCanWrite = true;			

							if( !m_nTxPending )
							{
								if( m_bSendPending ){
									m_bSendPending = false;
									DWORD dw; WSAIoctl( m_socket, SIO_FLUSH, NULL, 0, NULL, 0, &dw, NULL, NULL ); 
								}

								if( m_bWriteClosePending ){
									m_bWriteClosePending = false;
									shutdown( m_socket, SD_SEND );
									m_bLocalFinished	= true;
								}				
							}

						}// End if-else( no write errors )
					}// End if( FD_WRITE )
				}// End if( !Finished && IsConnected )												

				#else	/** Non-Windows **/

				timeval timeout; timeout.tv_sec = 0; timeout.tv_usec = 0;

				fd_set writable;
				FD_ZERO(&writable);
				FD_SET(m_socket, &writable);					
				if (select(m_socket+1, nullptr, &writable, nullptr, &timeout) == SOCKET_ERROR)
					throw TCPSocketException(S("Unable to poll socket status: "), GetLastSocketError());				

				if (m_State == States::Connecting)
				{										
					if (FD_ISSET(m_socket, &writable))
					{
						int ConnectionErrorCode = EINVAL;
						socklen_t optlen = sizeof(ConnectionErrorCode);
						if (getsockopt(m_socket, SOL_SOCKET, SO_ERROR, &ConnectionErrorCode, &optlen) == SOCKET_ERROR)
							throw TCPSocketException(S("Unable to retrieve connection status: "), GetLastSocketError());
						if (sizeof(ConnectionErrorCode) != optlen) 
							throw TCPSocketException(S("Unable to retrieve connection status: Argument size mismatch."));
						if (ConnectionErrorCode == 0)
						{
							m_State = States::Connected;
							m_bCanWrite = true;
						}
						else
							throw TCPSocketException(S("Error while attempting to connect: "), ConnectionErrorCode);
					}
				}
				else if (m_State != States::Disconnected)
				{
					if (!m_bLocalFinished && FD_ISSET(m_socket, &writable))
					{
						if (m_nTxPending)
						{
							assert (m_pTxBuffer);

							if (send(m_socket, (const char *)m_pTxBuffer, m_nTxPending, 0) == SOCKET_ERROR){
								int status = GetLastSocketError();
								if (WouldBlock(status)) m_bCanWrite = false;
								else OnSendError(status);
							} else {
								m_nTxPending = 0;
								m_bCanWrite = true;				
							}
						}
						else m_bCanWrite = true;

						if (!m_nTxPending)
						{
							if (m_bWriteClosePending){
								m_bWriteClosePending = false;
								shutdown( m_socket, SHUT_WR );
								m_bLocalFinished	= true;
							}
						}
					}
				}

				#endif

				/** Sockets have a messy interface for detecting EOF, the state where the other end of the pipe
					has performed a write-shutdown.  The way to detect it is to call recv() and watch for a zero
					return value (similar to the read() call for file system, a zero indicates EOF).  However,
					this has some problems in a network stream situation:

					1. There might be no bytes waiting, or fewer than requested in the recv() call, but the other end 
					has not actually closed the file.  This would cause recv() to return EAGAIN or EWOULDBLOCK for a 
					non-blocking socket.  A blocking socket would block in this case.

					2. We can detect the number of bytes waiting with an ioctl(...FIONREAD...) call.  Unlike recv(),
					however, this gives no indication of EOF and would actually return zero in that case, the same as
					if nothing is waiting.

					3. The select() call's receive set will be flagged whenever any incoming data is waiting, including 
					both content and the EOF indicator.  However, it does not indicate whether it is incoming data or
					an EOF indicator.

					Although we might be able to detect EOF by checking both items #2 and #3, the alternative is a
					receive buffer.  This seems like the most portable solution and it is used here.  However, there
					is an unhandled case mentioned in Windows documentation (see WSAEventSelect) where recv() can return
					the WSAEWOULDBLOCK code even when data is possibly available.  They aren't clear whether that can
					happen when ioctlsocket(...FIONREAD...) returns non-zero data, so I will assume that it's ok for
					the Get() calls that expect N bytes, but this is not verified and is not documented as reliable.
					There does not appear to be any better method to provide this interface.
				**/

				if (!m_bRemoteFinished && IsConnected())
				{
					if (m_RxBuffer.GetAvailable() > 0)
					{
						uint nAvailable = 0;
						void *pBuffer = m_RxBuffer.DirectAdd(nAvailable);
						// Note that ordinarily a user of DirectAdd() would repeat the operation a second time in case
						// a buffer wrap-around applied and the available space was "split".  However, this processing
						// function will be called routinely so it is unnecessary.
						int nBytes = recv(m_socket, (char *)pBuffer, nAvailable, 0);
						if (nBytes == SOCKET_ERROR){ 
							int ErrorCode = GetLastSocketError();
							if (!WouldBlock(ErrorCode)) OnRecvError( GetLastSocketError() ); 
						}
						else if (nBytes == 0)
						{
							/** EOF condition detected **/
							m_bRemoteFinished = true;
						}
						else m_RxBuffer.AddedDirect(nBytes);
					}
				}

				if (m_bLocalFinished && m_bRemoteFinished && m_RxBuffer.GetLength() == 0) m_State = States::Disconnected;
			}

			inline void TCPSocket::OnSendError( int nErrorCode )
			{
				BeginDisconnect();
				throw TCPSocketException(S("Transmission error on TCP Socket: "), GetLastSocketError(), TCPSocketException::g_SendErrors);				
			}

			inline void TCPSocket::OnRecvError( int nErrorCode )
			{
				BeginDisconnect();	
				throw TCPSocketException(S("Reception error on TCP Socket: "), GetLastSocketError(), TCPSocketException::g_RecvErrors);				
			}

			inline void TCPSocket::Add(byte b)
			{ 
				assert( !m_nTxPending );				// Assertion: Attempt to Add() more than once w/o checking for availability resulting in data loss.

				int status = send( m_socket, (const char *)&b, 1, 0 );
				if (status == SOCKET_ERROR){
					status = GetLastSocketError();
					if (WouldBlock(status)){
						m_bCanWrite = false;
						if (m_nTxAlloc < 1){
							if( m_pTxBuffer ){ delete[] m_pTxBuffer; m_pTxBuffer = NULL; }
							m_pTxBuffer = new byte [1024];
							m_nTxAlloc = 1024;
						}
						m_nTxPending = 1;
						m_pTxBuffer[0] = b;
						return;
					}
					else OnSendError( GetLastSocketError() );
				}
			}
			inline void TCPSocket::Add(void *pBlock, Int32 nLength)
			{ 
				assert( !m_nTxPending );				// Assertion: Attempt to Add() more than once w/o checking for availability resulting in data loss.

				int status = send( m_socket, (const char *)pBlock, nLength, 0 );
				if (status == SOCKET_ERROR){
					status = GetLastSocketError();
					if (WouldBlock(status)){
						m_bCanWrite = false;
						if (m_nTxAlloc < (UInt32)nLength){
							if( m_pTxBuffer ){ 
								delete[] m_pTxBuffer; 
								m_pTxBuffer = NULL; 
							}
							m_nTxAlloc = MaxOf((UInt32)1024, (UInt32)nLength);
							m_pTxBuffer = new byte [m_nTxAlloc];
						}
						m_nTxPending = nLength;
						CopyMemory( m_pTxBuffer, pBlock, nLength );
						return;
					}
					else OnSendError( GetLastSocketError() );
				}
			}
			inline void TCPSocket::Add(const char* lpsz)
			{
				size_t	nLength	= strlen(lpsz);
				if (nLength > Int32_MaxValue) throw ArgumentOutOfRangeException();
				Add( (void *)lpsz, (int)nLength );
			}

			inline void TCPSocket::Add(const string& str)
			{	
				if (str.length() > Int32_MaxValue) throw ArgumentOutOfRangeException();
				Add( (void *)str.c_str(), (int)str.length() );
			}

			inline void TCPSocket::Get(byte& b)
			{
				if (m_RxBuffer.GetLength())
				{
					m_RxBuffer.Get(b);
					return;
				}

				int nBytes = recv(m_socket, (char *)&b, 1, 0);
				if (nBytes == SOCKET_ERROR) OnRecvError( GetLastSocketError() );
				else if (nBytes != 1) throw TCPSocketException(S("Expected single-byte recv."));
			}
			inline void TCPSocket::Get(void *pBlock, UInt32 dwLength)
			{ 
				if (dwLength > Int32_MaxValue) throw NotSupportedException("Exceeded TCPSocket::Get() length limit.");
				uint FromBuffer = MinOf((UInt32)m_RxBuffer.GetLength(), dwLength);
				if (FromBuffer > 0)
				{
					m_RxBuffer.Get(pBlock, FromBuffer);
				}
				dwLength -= FromBuffer;
				assert(dwLength == 0 || m_RxBuffer.GetLength() == 0);			// We didn't completely empty out the buffer.
				if (dwLength > 0)
				{
					int nBytes = recv(m_socket, (char *)pBlock + FromBuffer, dwLength, 0);
					if (nBytes == SOCKET_ERROR) OnRecvError( GetLastSocketError() );
					else if (nBytes != (int)dwLength) throw TCPSocketException(S("Expected buffered receive data."));
				}
			}

			inline void TCPSocket::Flush(){ 
				if( m_nTxPending ) m_bSendPending = true;
				#ifdef _WINDOWS
				else {
					m_bSendPending = false;
					DWORD dw; WSAIoctl( m_socket, SIO_FLUSH, NULL, 0, NULL, 0, &dw, NULL, NULL ); 
				}
				#endif
			}

			inline uint TCPSocket::GetRxCount()
			{ 			
					// ioctlsocket( FIONREAD ) gives us the number of bytes which can be read in a single recv() call.
				#ifdef _WINDOWS
				u_long lSize = 0;
				if( ioctlsocket( m_socket, FIONREAD, &lSize ) == SOCKET_ERROR )
				#else
				UInt32 lSize = 0;
				if( ioctl( m_socket, FIONREAD, &lSize ) == SOCKET_ERROR )
				#endif
				{
					OnRecvError( GetLastSocketError() );
					return false;	
				}
				return lSize + m_RxBuffer.GetLength();
			}

			inline bool TCPSocket::IsRxReady(){ return GetRxCount() > 0; }

			inline bool TCPSocket::IsRemoteFinished(){ return (m_State == States::Disconnected) || m_bRemoteFinished; }
			inline bool TCPSocket::IsLocalFinished(){ return (m_State == States::Disconnected) || m_bLocalFinished; }			
		}
	}
}

#endif	// __TCPSocket_h__

//	End of TCPSocket.h



