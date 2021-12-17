/////////
//	SocketExceptions.h
//  Copyright (C) 2014 by Wiley Black
////

#ifndef __SocketExceptions_h__
#define __SocketExceptions_h__

/** Table of contents and references **/

namespace wb
{
	namespace net
	{		
		namespace sockets
		{
			class SocketException;
			class TCPSocketException;
			class UDPSocketException;

			class TCPServerSocket;
			class TCPSocket;

			class UDPSocket;
		}
	}
};

#include "wbFoundation.h"

namespace wb
{
	namespace net
	{
		namespace sockets
		{
			using namespace wb;			

			class SocketException : public NetworkException
			{
			public:
				struct	WSAErrorReport	{ int	nErrorCode;		const char *pMessage;	};						
				
				static const WSAErrorReport	g_Errors[];				

				SocketException(int nErrorCode, const WSAErrorReport *pTable = g_Errors);
				SocketException(const char* pszMessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors);
				SocketException(const string& MessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors);

			public:
				SocketException() : NetworkException("A socket error has occurred.") { }
				SocketException(const char * const &message) : NetworkException(message) { }
				SocketException(const string& message) : NetworkException(message) { }
				SocketException(const SocketException &right) : NetworkException(right) { }
				SocketException(SocketException&& from) : NetworkException(from) { }
				SocketException& operator=(const SocketException &right) { NetworkException::operator=(right); return *this; }
			};

			class TCPSocketException : public SocketException
			{
				friend class TCPServerSocket;
				friend class TCPSocket;				

				static const WSAErrorReport	g_ConnectionErrors[];
				static const WSAErrorReport	g_SetOptErrors[];
				static const WSAErrorReport g_BindErrors[];
				static const WSAErrorReport	g_AcceptErrors[];
				static const WSAErrorReport	g_SendErrors[];
				static const WSAErrorReport	g_RecvErrors[];

				TCPSocketException(int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(nErrorCode,pTable) { }
				TCPSocketException(const char* pszMessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(pszMessagePrefix,nErrorCode,pTable) { }
				TCPSocketException(const string& MessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(MessagePrefix,nErrorCode,pTable) { }
			public:
				TCPSocketException() : SocketException() { }
				TCPSocketException(const char * const &message) : SocketException(message) { }
				TCPSocketException(const string& message) : SocketException(message) { }
				TCPSocketException(const TCPSocketException &right) : SocketException(right) { }
				TCPSocketException(TCPSocketException&& from) : SocketException(from) { }
				TCPSocketException& operator=(const TCPSocketException &right) { SocketException::operator=(right); return *this; }
			};

			class UDPSocketException : public SocketException
			{
				friend class UDPSocket;
			
				static const WSAErrorReport	g_IOCtlErrors[];
				static const WSAErrorReport	g_RecvErrors[];
				static const WSAErrorReport g_SendErrors[];

				UDPSocketException(int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(nErrorCode,pTable) { ErrorCode = nErrorCode; }
				UDPSocketException(const char* pszMessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(pszMessagePrefix,nErrorCode,pTable) { ErrorCode = nErrorCode; }
				UDPSocketException(const string& MessagePrefix, int nErrorCode, const WSAErrorReport *pTable = g_Errors) : SocketException(MessagePrefix,nErrorCode,pTable) { ErrorCode = nErrorCode; }
			public:
				int ErrorCode;				// Non-zero if a system level error.  errno value on Linux, WSAGetLastError() value on Windows.

				UDPSocketException() : SocketException() { ErrorCode=0; }
				UDPSocketException(const char * const &message) : SocketException(message) { ErrorCode=0; }
				UDPSocketException(const string& message) : SocketException(message) { ErrorCode=0; }
				UDPSocketException(const UDPSocketException &right) : SocketException(right) { ErrorCode = right.ErrorCode; }
				UDPSocketException(UDPSocketException&& from) : SocketException(from) { ErrorCode = from.ErrorCode; }
				UDPSocketException& operator=(const UDPSocketException &right) { SocketException::operator=(right); ErrorCode = right.ErrorCode; return *this; }
			};

			/** Inline Methods **/
			
			#ifdef _LINUX
			   
			inline SocketException::SocketException(int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				// ThrowFromErrno() is a little inconvenient because it actually throws the exception that we want.  So we'll
				// throw and catch it, then transfer the message to here.  This captures the exception into a SocketException.				
				try { Exception::ThrowFromErrno(nErrorCode); }
				catch (std::exception& ex) { Message = ex.what(); }
			}

			inline SocketException::SocketException(const char* pszMessagePrefix, int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				try { Exception::ThrowFromErrno(nErrorCode); }
				catch (std::exception& ex) { 					
					Message = string(pszMessagePrefix) + ex.what(); 
				}
			}

			inline SocketException::SocketException(const string& MessagePrefix, int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				try { Exception::ThrowFromErrno(nErrorCode); }
				catch (std::exception& ex) { 
					Message = MessagePrefix + ex.what();
				}
			}			

			#ifdef PrimaryModule
			/*static*/ const SocketException::WSAErrorReport	SocketException::g_Errors[] = { };

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_ConnectionErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_SetOptErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_BindErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_AcceptErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_SendErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_RecvErrors[] = { };			

			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_IOCtlErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_RecvErrors[] = { };
			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_SendErrors[] = { };			
			#endif

			#else		// Windows

			inline SocketException::SocketException(int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				for(int ii=0; pTable[ii].pMessage != nullptr; ii++)
				{
					if (pTable[ii].nErrorCode == nErrorCode) { Message = pTable[ii].pMessage; return; }				
				}
			}

			inline SocketException::SocketException(const char* pszMessagePrefix, int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				for(int ii=0; pTable[ii].pMessage != nullptr; ii++)
				{
					if (pTable[ii].nErrorCode == nErrorCode) { Message = string(pszMessagePrefix) + pTable[ii].pMessage; return; }				
				}
				Message = string(pszMessagePrefix) + S("Unknown socket error.");
			}

			inline SocketException::SocketException(const string& MessagePrefix, int nErrorCode, const WSAErrorReport *pTable)
				: NetworkException()
			{
				for(int ii=0; pTable[ii].pMessage != nullptr; ii++)
				{
					if (pTable[ii].nErrorCode == nErrorCode) { Message = MessagePrefix + pTable[ii].pMessage; return; }				
				}
				Message = MessagePrefix + S("Unknown socket error.");
			}

			/////////
			//	Detailed Error Reports
			//

			#ifdef PrimaryModule

			/** General socket errors **/

			/*static*/ const SocketException::WSAErrorReport	SocketException::g_Errors[28] = {

				{ WSAEACCES, S("The requested address is a broadcast address, but the appropriate flag was not set. Call setsockopt with the SO_BROADCAST parameter to allow the use of the broadcast address. ") },
				{ WSAEADDRINUSE, S("A process on the machine is already bound to the same fully-qualified address and the socket has not been marked to allow address reuse with SO_REUSEADDR. For example, the IP address and port are bound in the af_inet case). (See the SO_REUSEADDR socket option under setsockopt.) ") },
				{ WSAEADDRNOTAVAIL, S("The specified address is not a valid address for this machine. ") },
				{ WSAEAFNOSUPPORT, S("Addresses in the specified family cannot be used with this socket.") }, 
				{ WSAEALREADY, S("A nonblocking connect call is in progress on the specified socket.") },
				{ WSAECONNABORTED, S("The virtual circuit was terminated due to a time-out or other failure. ") },
				{ WSAECONNRESET, S("The virtual circuit was reset by the remote side executing a \"hard\" or \"abortive\" close. ") },
				{ WSAECONNREFUSED, S("The attempt to connect was forcefully rejected.") },
				{ WSAEFAULT, S("The buf parameter is not completely contained in a valid part of the user address space. ") },
				{ WSAEHOSTUNREACH, S("The remote host cannot be reached from this host at this time. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAEINTR, S("A blocking Windows Sockets 1.1 call was canceled through WSACancelBlockingCall. ") },
				{ WSAEINVAL, S("The socket has not been bound with bind, or an unknown flag was specified, or MSG_OOB was specified for a socket with SO_OOBINLINE enabled. ") },
				{ WSAEISCONN, S("The socket is already connected.") },
				{ WSAEMFILE, S("The queue is nonempty upon entry to accept and there are no descriptors available.") },
				{ WSAEMSGSIZE, S("The socket is message oriented, and the message is larger than the maximum supported by the underlying transport. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAENETRESET, S("The connection has been broken due to the \"keep-alive\" activity detecting a failure while the operation was in progress. ") },
				{ WSAENETUNREACH, S("The network cannot be reached from this host at this time.") },
				{ WSAENOBUFS, S("No buffer space is available. ") },
				{ WSAENOTCONN, S("The socket is not connected. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ WSAEOPNOTSUPP, S("Operation not supported. ") },
				{ WSAESHUTDOWN, S("The socket has been shut down; it is not possible to send on a socket after shutdown has been invoked with how set to SD_SEND or SD_BOTH. ") },
				{ WSAETIMEDOUT, S("The connection has been dropped, because of a network failure or because the system on the other end went down without notice. ") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the requested operation would block. ") },
				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ 0, NULL }

			};

			/** TCP Detailed Errors **/

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_ConnectionErrors[19] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function.") },
				{ WSAENETDOWN, S("The network subsystem has failed.") },
				{ WSAEADDRINUSE, S("The socket's local address is already in use and the socket was not marked to allow address reuse with SO_REUSEADDR. This error usually occurs when executing bind, but could be delayed until this function if the bind was to a partially wild-card address (involving ADDR_ANY) and if a specific address needs to be committed at the time of this function. ") },
				{ WSAEINTR, S("The (blocking) Windows Socket 1.1 call was canceled through WSACancelBlockingCall.") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function.") },
				{ WSAEALREADY, S("A nonblocking connect call is in progress on the specified socket.") },
				{ WSAEADDRNOTAVAIL, S("The remote address is not a valid address (such as ADDR_ANY).") },
				{ WSAEAFNOSUPPORT, S("Addresses in the specified family cannot be used with this socket.") }, 
				{ WSAECONNREFUSED, S("The attempt to connect was forcefully rejected.") },
				{ WSAEFAULT, S("The name or the namelen parameter is not a valid part of the user address space, the namelen parameter is too small, or the name parameter contains incorrect address format for the associated address family.") },
				{ WSAEINVAL, S("The parameter s is a listening socket, or the destination address specified is not consistent with that of the constrained group the socket belongs to.") },
				{ WSAEISCONN, S("The socket is already connected (connection-oriented sockets only).") },
				{ WSAENETUNREACH, S("The network cannot be reached from this host at this time.") },
				{ WSAENOBUFS, S("No buffer space is available. The socket cannot be connected.") },
				{ WSAENOTSOCK, S("The descriptor is not a socket.") },
				{ WSAETIMEDOUT, S("Attempt to connect timed out without establishing a connection.") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the connection cannot be completed immediately.") },
				{ WSAEACCES, S("Attempt to connect datagram socket to broadcast address failed because setsockopt option SO_BROADCAST is not enabled.") },
				{ 0, NULL }

			};

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_SetOptErrors[10] = {
				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEFAULT, S("optval is not in a valid part of the process address space or optlen parameter is too small. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAEINVAL, S("level is not valid, or the information in optval is not valid. ") },
				{ WSAENETRESET, S("Connection has timed out when SO_KEEPALIVE is set. ") },
				{ WSAENOPROTOOPT, S("The option is unknown or unsupported for the specified provider or socket (see SO_GROUP_PRIORITY limitations). ") },
				{ WSAENOTCONN, S("Connection has been reset when SO_KEEPALIVE is set. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ 0, NULL }
			};

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_BindErrors[11] = {
				{ WSANOTINITIALISED, S("A successful WSAStartup call must occur before using this function.") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEACCES, S("Access Error") },
				{ WSAEADDRINUSE, S("A process on the machine is already bound to the same fully-qualified address and the socket has not been marked to allow address reuse with SO_REUSEADDR. For example, the IP address and port are bound in the af_inet case). (See the SO_REUSEADDR socket option under setsockopt.) ") },
				{ WSAEADDRNOTAVAIL, S("The specified address is not a valid address for this machine. ") },
				{ WSAEFAULT, S("The name or namelen parameter is not a valid part of the user address space, the namelen parameter is too small, the name parameter contains an incorrect address format for the associated address family, or the first two bytes of the memory block specified by name does not match the address family associated with the socket descriptor s. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAEINVAL, S("The socket is already bound to an address. ") },
				{ WSAENOBUFS, S("Not enough buffers available, too many connections. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ 0, NULL }

			};

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_AcceptErrors[12] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this FUNCTION.") },
				{ WSAENETDOWN, S("The network subsystem has failed.") },
				{ WSAEFAULT, S("The addrlen parameter is too small or addr is not a valid part of the user address space.") },
				{ WSAEINTR, S("A blocking Windows Sockets 1.1 call was canceled through WSACancelBlockingCall.") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function.") },
				{ WSAEINVAL, S("The listen function was not invoked prior to accept.") },
				{ WSAEMFILE, S("The queue is nonempty upon entry to accept and there are no descriptors available.") },
				{ WSAENOBUFS, S("No buffer space is available.") },
				{ WSAENOTSOCK, S("The descriptor is not a socket.") },
				{ WSAEOPNOTSUPP, S("The referenced socket is not a type that supports connection-oriented service.") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and no connections are present to be accepted.") },
				{ 0, NULL }

			};

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_SendErrors[21] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEACCES, S("The requested address is a broadcast address, but the appropriate flag was not set. Call setsockopt with the SO_BROADCAST parameter to allow the use of the broadcast address. ") },	
				{ WSAEINTR, S("A blocking Windows Sockets 1.1 call was canceled through WSACancelBlockingCall. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },	
				{ WSAEFAULT, S("The buf parameter is not completely contained in a valid part of the user address space. ") },
				{ WSAENETRESET, S("The connection has been broken due to the \"keep-alive\" activity detecting a failure while the operation was in progress. ") },
				{ WSAENETUNREACH, S("The network cannot be reached from this host at this time.") },
				{ WSAENOBUFS, S("No buffer space is available. ") },
				{ WSAENOTCONN, S("The socket is not connected. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ WSAEOPNOTSUPP, S("MSG_OOB was specified, but the socket is not stream-style such as type SOCK_STREAM, out-of-band data is not supported in the communication domain associated with this socket, or the socket is unidirectional and supports only receive operations. ") },
				{ WSAESHUTDOWN, S("The socket has been shut down; it is not possible to send on a socket after shutdown has been invoked with how set to SD_SEND or SD_BOTH. ") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the requested operation would block. ") },
				{ WSAEMSGSIZE, S("The socket is message oriented, and the message is larger than the maximum supported by the underlying transport. ") },
				{ WSAEHOSTUNREACH, S("The remote host cannot be reached from this host at this time. ") },
				{ WSAEINVAL, S("The socket has not been bound with bind, or an unknown flag was specified, or MSG_OOB was specified for a socket with SO_OOBINLINE enabled. ") },
				{ WSAECONNABORTED, S("The virtual circuit was terminated due to a time-out or other failure. ") },
				{ WSAECONNRESET, S("The virtual circuit was reset by the remote side executing a \"hard\" or \"abortive\" close. ") },
				{ WSAETIMEDOUT, S("The connection has been dropped, because of a network failure or because the system on the other end went down without notice. ") },
				{ 0, NULL }

			};

			/*static*/ const SocketException::WSAErrorReport	TCPSocketException::g_RecvErrors[17] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEFAULT, S("The buf parameter is not completely contained in a valid part of the user address space. ") },
				{ WSAENOTCONN, S("The socket is not connected. ") },
				{ WSAEINTR, S("The (blocking) call was canceled through WSACancelBlockingCall. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAENETRESET, S("The connection has been broken due to the keep-alive activity detecting a failure while the operation was in progress. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ WSAEOPNOTSUPP, S("MSG_OOB was specified, but the socket is not stream-style such as type SOCK_STREAM, out-of-band data is not supported in the communication domain associated with this socket, or the socket is unidirectional and supports only send operations. ") },
				{ WSAESHUTDOWN, S("The socket has been shut down; it is not possible to recv on a socket after shutdown has been invoked with how set to SD_RECEIVE or SD_BOTH. ") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the receive operation would block. ") },
				{ WSAEMSGSIZE, S("The message was too large to fit into the specified buffer and was truncated. ") },
				{ WSAEINVAL, S("The socket has not been bound with bind, or an unknown flag was specified, or MSG_OOB was specified for a socket with SO_OOBINLINE enabled or (for byte stream sockets only) len was zero or negative. ") },
				{ WSAECONNABORTED, S("The virtual circuit was terminated due to a time-out or other failure. ") },
				{ WSAETIMEDOUT, S("The connection has been dropped because of a network failure or because the peer system failed to respond. ") },
				{ WSAECONNRESET, S("The virtual circuit was reset by the remote side executing a \"hard\" or \"abortive\" close. ") },
				{ 0, NULL }

			};			

			/** UDP Detailed Errors **/

			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_IOCtlErrors[6] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAENOTSOCK, S("The descriptor s is not a socket. ") },
				{ WSAEFAULT, S("The argp parameter is not a valid part of the user address space. ") },
				{ 0, NULL }

			};


			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_RecvErrors[17] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function. ") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEFAULT, S("The buf parameter is not completely contained in a valid part of the user address space. ") },
				{ WSAENOTCONN, S("The socket is not connected. ") },
				{ WSAEINTR, S("The (blocking) call was canceled through WSACancelBlockingCall. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAENETRESET, S("The connection has been broken due to the keep-alive activity detecting a failure while the operation was in progress. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ WSAEOPNOTSUPP, S("MSG_OOB was specified, but the socket is not stream-style and out-of-band data is not supported in the communication domain associated with this socket, or the socket is unidirectional and supports only send operations. ") },
				{ WSAESHUTDOWN, S("The socket has been shut down; it is not possible to recv on a socket after shutdown has been invoked with how set to SD_RECEIVE or SD_BOTH. ") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the receive operation would block. ") },
				{ WSAEMSGSIZE, S("The message was too large to fit into the specified buffer and was truncated. ") },
				{ WSAEINVAL, S("The socket has not been bound with bind, or an unknown flag was specified. ") },
				{ WSAECONNABORTED, S("The virtual circuit was terminated due to a time-out or other failure. ") },
				{ WSAETIMEDOUT, S("The connection has been dropped because of a network failure or because the peer system failed to respond. ") },
				{ WSAECONNRESET, S("A previous send operation resulted in an ICMP \"Port Unreachable\" message. ") },
				{ 0, NULL }

			};

			/*static*/ const SocketException::WSAErrorReport	UDPSocketException::g_SendErrors[20] = {

				{ WSANOTINITIALISED, S("A successful WSAStartup must occur before using this function.") },
				{ WSAENETDOWN, S("The network subsystem has failed. ") },
				{ WSAEACCES, S("The requested address is a broadcast address, but the appropriate flag was not set. Call setsockopt with the SO_BROADCAST parameter to allow the use of the broadcast address. ") },
				{ WSAEINTR, S("A blocking Windows Sockets 1.1 call was canceled through WSACancelBlockingCall. ") },
				{ WSAEINPROGRESS, S("A blocking Windows Sockets 1.1 call is in progress, or the service provider is still processing a callback function. ") },
				{ WSAEFAULT, S("The buf parameter is not completely contained in a valid part of the user address space. ") },
				{ WSAENETRESET, S("The connection has been broken due to the 'keep-alive' activity detecting a failure while the operation was in progress. ") },
				{ WSAENOBUFS, S("No buffer space is available. ") },
				{ WSAENOTCONN, S("The socket is not connected. ") },
				{ WSAENOTSOCK, S("The descriptor is not a socket. ") },
				{ WSAEOPNOTSUPP, S("MSG_OOB was specified, but the socket is not stream-style such as type SOCK_STREAM, out-of-band data is not supported in the communication domain associated with this socket, or the socket is unidirectional and supports only receive operations. ") },
				{ WSAESHUTDOWN, S("The socket has been shut down; it is not possible to send on a socket after shutdown has been invoked with how set to SD_SEND or SD_BOTH. ") },
				{ WSAEWOULDBLOCK, S("The socket is marked as nonblocking and the requested operation would block. ") },
				{ WSAEMSGSIZE, S("The socket is message oriented, and the message is larger than the maximum supported by the underlying transport. ") },
				{ WSAEHOSTUNREACH, S("The remote host cannot be reached from this host at this time. ") },
				{ WSAEINVAL, S("The socket has not been bound with bind, or an unknown flag was specified, or MSG_OOB was specified for a socket with SO_OOBINLINE enabled. ") },
				{ WSAECONNABORTED, S("The virtual circuit was terminated due to a time-out or other failure. The application should close the socket as it is no longer usable. ") },
				{ WSAECONNRESET, S("The virtual circuit was reset by the remote side executing a 'hard' or 'abortive' close. For UPD sockets, the remote host was unable to deliver a previously sent UDP datagram and responded with a 'Port Unreachable' ICMP packet. The application should close the socket as it is no longer usable. ") },
				{ WSAETIMEDOUT, S("The connection has been dropped, because of a network failure or because the system on the other end went down without notice. ") },
				{ 0, NULL }
			};			

			#endif		// PrimaryModule

			#endif		// _WINDOWS
		}
	}
}

#endif

//	End of SocketExceptions.h
