/**
 *  XMLSocket.h
 *	Copyright (C) 2021-2022 by Wiley Black
 *
 *  Provides the xmlsocket namespace with Protocol and Server classes.  The Protocol class provides Send() and Receive() functions implementing the simple XMLSocket 
 *	protocol for transferring XML-based messages across a TCP/IP socket.  The Server class provides a thread that listens for incoming socket connections and services 
 *  each on an independent thread, providing a Protocol object that for each specific connection.  The XMLSocket protocol provides no interpretation of messages beyond 
 *	XML parsing and implements no security at this level.
 */

#ifndef __XMLSocket_h__
#define __XMLSocket_h__

#include "../wbCore.h"
#include <iostream>
#include <queue>

using namespace wb::net;
using namespace wb::net::sockets;
using namespace wb::sys::threading;
using namespace wb::xml;
using namespace std;

namespace wb { namespace net { namespace xmlsocket {

// Forward declarations
class Server;

/// <summary>
/// The Protocol class implements the heart of the XMLSocket.  It provides Send() and Receive() functions.  The Send() function accepts either an XmlElement or an XML
/// formatted string.  The Receive() method can be used to retrieve an XML message from the socket.  For client-side use (contacting a server), call the 
/// Protocol::Client() static function and provide the server address and port that you want to contact.  For server-side use, see the Server class.  Internally,
/// the Protocol class sends each message as a 4-byte prefix specifying the length of the message followed by the message content.  The message content is then
/// parsed as an XML message.  The format of the XML message is left up to the caller.  
/// </summary>
class Protocol
{
	std::unique_ptr<TCPSocket>		_pConnection;
	bool							_SendDisconnect;
	bool							_RxPrefixReceived;
	UInt32							_RxMsgLen;
	io::MemoryStream				_Buffer;
	XmlParser						_Parser;
	
	void PrefixSend(void* pMessage, UInt32 nLength)
	{
		UInt32 nlen = (UInt32)htonl((UInt32)nLength);
		pSocket->Add(&nlen, sizeof(UInt32));
		pSocket->Add(pMessage, nLength);
		pSocket->Flush();
	}

	friend class Server;

	Protocol(std::unique_ptr<TCPSocket>&& socket, bool SendDisconnect = false)
		: pSocket(std::move(socket)), _SendDisconnect(SendDisconnect)
	{
		_RxPrefixReceived = false;
		_RxMsgLen = 0;
		_Buffer.EnsureCapacity(65536);
	}

public:
	std::unique_ptr<TCPSocket>		pSocket;

	~Protocol()
	{
		if (pSocket != nullptr)
		{
			if (_SendDisconnect) Send("<Disconnect />");
			pSocket->BeginDisconnect();
			pSocket->Process();
			pSocket = nullptr;
		}
	}

	/// <summary>
	/// Call Client() to construct a new Protocol object for use by a client in contacting a server.  This overload is useful if the caller
	/// wants to avoid the blocking behavior of the other overload of Client().  To use this overload, the socket must already be connected
	/// before calling Client().
	/// </summary>
	/// <param name="socket">An already-connected socket.</param>
	/// <param name="SendDisconnect">True if the client should transmit a "&lt;Disconnect/&gt;" message when the Protocol object is
	/// destroyed.</param>
	static Protocol Client(std::unique_ptr<TCPSocket>&& socket, bool SendDisconnect = false)
	{
		return Protocol(std::move(socket), SendDisconnect);
	}

	/// <summary>
	/// Call Client() to construct a new Protocol object for use by a client in contacting a server.  This overload blocks until the connection
	/// is established or an error occurs.
	/// </summary>	
	static Protocol Client(sockets::Provider& Provider, IP4Address RemoteHost, bool SendDisconnect = false)
	{
		auto pSocket = make_unique<TCPSocket>(Provider.CreateTCPSocket());
		IP4Address	addrSource(IP4Address::ANY, IP4Address::ANY);
		pSocket->BeginConnect(RemoteHost, addrSource);
		while (!pSocket->IsConnected()) {
			pSocket->Process(); Thread::Yield();
		}
		return Protocol(std::move(pSocket), SendDisconnect);
	}
	
	bool IsConnected() { return pSocket->IsConnected(); }
	bool IsLocalFinished() { return pSocket->IsLocalFinished(); }
	bool IsRemoteFinished() { return pSocket->IsRemoteFinished(); }	
	void BeginDisconnect() { pSocket->BeginDisconnect(); }
	wb::net::sockets::IP4Address GetDestinationAddress() { return pSocket->GetDestinationAddress(); }

	/// <summary>
	/// Send the provided message, which must contain valid XML, to the remote host of this socket.
	/// </summary>	
	void Send(const std::string& XmlMessage)
	{
		PrefixSend((void*)XmlMessage.c_str(), (UInt32)XmlMessage.length());
	}

	/// <summary>
	/// Send the provided XML message to the remote host of this socket.
	/// </summary>	
	void Send(XmlElement& Message)
	{
		Send(Message.ToString());
	}
	
	enum { InfiniteTime = -1 };

	/// <summary>
	/// Retrieves a message from the remote host on this socket connection.  If TimeoutInMilliseconds
	/// is InfiniteTime (default), then Receive() blocks until a complete message is received or a
	/// socket error occurs (such as a disconnected socket), which will result in the exception being
	/// thrown from Receive() as well.  If TimeoutInMilliseconds is given as 0, then Receive() polls
	/// for a waiting message and then returns immediately.  If TimeoutInMilliseconds is non-zero,
	/// then Receive() blocks for the duration of the timeout and then throws a TimeoutException if
	/// no message has been received.
	/// </summary>	
	/// <returns>The XML message received and parsed if received.  If TimeoutInMilliseconds is
	/// specified as zero and no message is available, Receive() will return nullptr instead.</returns>
	std::unique_ptr<XmlDocument>	Receive(int TimeoutInMilliseconds = InfiniteTime)
	{
		if (!pSocket) throw Exception("Receive() failed because no connection was established.");

		wb::Stopwatch sw = wb::Stopwatch::StartNew();
		// Receive 4-byte prefix...
		if (!_RxPrefixReceived)
		{
			pSocket->Process();
			while (pSocket->GetRxCount() < sizeof(UInt32))
			{				
				if (TimeoutInMilliseconds == 0) return nullptr;
				if (sw.GetElapsedMilliseconds() >= TimeoutInMilliseconds) throw TimeoutException("Receive() failed while waiting for response from server (" + Convert::ToString(sw.GetElapsedMilliseconds() / 1000.0, 0, 2) + " seconds).");
				Thread::Yield();
				pSocket->Process();
			}

			pSocket->Get(&_RxMsgLen, sizeof(UInt32));			
			if (IsLittleEndian()) SwapEndian(_RxMsgLen);
			_RxPrefixReceived = true;
			if (_RxMsgLen >= _Buffer.GetCapacity()) throw FormatException("Received message prefix exceeds maximum length.");
		}

		// Receive message...
		for (;;)
		{
			pSocket->Process();
			uint Remaining = _RxMsgLen - _Buffer.GetLength();
			if (Remaining > 0)
			{
				uint Avail = pSocket->GetRxCount();
				if (Avail == 0)
				{
					if (TimeoutInMilliseconds == 0) return nullptr;
					if (sw.GetElapsedMilliseconds() >= TimeoutInMilliseconds) throw TimeoutException("Receive() failed while waiting for response from server (" + Convert::ToString(sw.GetElapsedMilliseconds() / 1000.0, 0, 2) + " seconds).");
					Thread::Yield();
					pSocket->Process();
				}
				if (Avail > Remaining) Avail = Remaining;
				pSocket->Get(_Buffer.GetDirectAccess(_Buffer.GetLength()), Avail);
				_Buffer.SetLength(_Buffer.GetLength() + Avail);
			}
			else break;
		}				
		
		_Buffer.Rewind();
		std::unique_ptr<XmlDocument> pDoc;
		try
		{
			pDoc = XmlParser::Parse(_Buffer, pSocket->GetDestinationAddress().ToString());
		}
		catch (std::exception& ex)
		{
			_Buffer.Rewind();
			throw Exception(string(ex.what()) + "\nWhile parsing incoming message (" + std::to_string(_Buffer.GetLength()) + " bytes):\n" + wb::io::StreamToString(_Buffer));
		}
		if (pDoc == nullptr) throw Exception("Expected XML-format message.");
		_Buffer.SetLength(0);
		return pDoc;		
	}	
};

/// <summary>
/// Implements a socket Server (listener) by spawning a new thread to monitor for incoming connections on the specified port and a thread pool for 
/// quickly and independently servicing incoming connections.  The Server class is abstract as the ProvideService() pure virtual function must be 
/// implemented for use.  The ProvideService() function services a new client connection while monitoring the "IsShuttingDown()" value to watch for
/// immediate termination.
/// </summary>
class Server
{
	unique_ptr<TCPServerSocket>				_pListener;	

	unique_ptr<Thread>						_pMainThread;
	wb::sys::threading::EventWaitHandle		_ShutdownEvent;
	vector<Thread>							_ServiceThreads;

	// Protected by the _CS lock:
	CriticalSection							_CS;
	queue<unique_ptr<TCPSocket>>			_ClientQueue;
	string									_Error;

	static UInt32 MainThreadLauncher(void* lpParameter)
	{
		return ((Server*)lpParameter)->MainThread();
	}

	UInt32 MainThread()
	{
		try
		{
			bool AcceptingConnections = true;
			while (!_ShutdownEvent.WaitOne(0))
			{
				if (AcceptingConnections)
				{
					auto pNewSocket = make_unique<TCPSocket>(_pListener->AcceptConnection());
					// AcceptConnection(), for servers, returns NULL if no incoming connections are pending...
					if (pNewSocket == nullptr) {
						Thread::Yield();
						continue;
					}

					{
						Lock lock(_CS);
						_ClientQueue.push(pNewSocket);
						// Check if we have a significant queue built up.  If so, let's stop accepting sockets and leave
						// additional queued items in the listener's queue.
						AcceptingConnections = (_ClientQueue.size() < _ServiceThreads.size());
					}
				}
				else
				{
					Thread::Yield();
					{
						Lock lock(_CS);
						AcceptingConnections = (_ClientQueue.size() < _ServiceThreads.size());
					}
				}
			}
			return 0;
		}
		catch (std::exception& ex)
		{
			// Note that an exception here on the main thread will override any other errors, i.e. from worker threads.
			Lock lock(_CS);
			_Error = ex.what();
			return -1;
		}		
	}

	static UInt32 ServiceThreadLauncher(void* lpParameter)
	{
		return ((Server*)lpParameter)->ServiceThread();
	}

	UInt32 ServiceThread()
	{
		try
		{
			while (!_ShutdownEvent.WaitOne(0))
			{
				unique_ptr<TCPSocket> pClientSocket;
				{
					Lock lock(_CS);
					if (!_ClientQueue.empty()) {
						pClientSocket = std::move(_ClientQueue.front());
						_ClientQueue.pop();
					}
				}
				if (pClientSocket == nullptr) {
					Thread::Yield();
					continue;
				}
				Protocol xmlproto(std::move(pClientSocket), false);
				ProvideService(xmlproto);
			}
		}
		catch (std::exception& ex)
		{
			// Note that an exception here on the main thread will override any other errors, i.e. from worker threads.
			Lock lock(_CS);
			if (_Error.length() == 0) _Error = ex.what();
			return -1;
		}
		return 0;
	}

protected:

	/// <summary>
	/// Call IsShuttingDown() from ProvideService() routinely to check if the server is being shut down.  If
	/// IsShuttingDown() returns true, then ProvideService() should abort all operations and return as
	/// quickly as practical.
	/// </summary>	
	bool IsShuttingDown() {
		return _ShutdownEvent.WaitOne(0);
	}

	/// <summary>
	/// ProvideService() is a pure virtual function that must be overridden in a derived class before the
	/// Server can be used.  It must monitor IsShuttingDown() as it services the client.  The new client
	/// is provided as an argument.  ProvideService() should not return until the client is completely
	/// serviced and the connection has been or can be closed or IsShuttingDown() returns true.  Note
	/// that ProvideService() is called from an independent thread out of the thread pool created by the
	/// Server.
	/// </summary>	
	virtual void ProvideService(Protocol& Client) = 0;

public:

	/// <summary>
	/// Construct a new XMLSocket Server.
	/// </summary>
	/// <param name="Provider">The sockets::Provider for TCP connections by the server.</param>
	/// <param name="Port">The port number to provide this service on.</param>
	/// <param name="MaxConnections">The maximum number of sockets to service
	/// simultaneously.  MaxConnections translates to the number of threads launched at
	/// construction, all of which start out idle until an incoming connection appears.  Additional
	/// connections are queued for processing after a thread becomes available.</param>
	/// <param name="Name">Friendly name of the server, used for diagnostics.</param>
	Server(sockets::Provider& Provider, int Port, int MaxConnections = 20, string Name = "XML Server")
		: 
		_ShutdownEvent(false, EventResetMode::ManualReset, to_osstring(Name + " Shutdown Event"))
	{
		IP4Address BindAddress(IP4Address::ANY, Port);
		_pListener = make_unique<TCPServerSocket>(Provider.CreateTCPServerSocket(BindAddress));

		// Start the listener thread...
		_pMainThread = make_unique<Thread>(new Thread(MainThreadLauncher, (void*)this));
		_pMainThread->Start();

		// Start the worker pool of threads, although they will all be initially idle...
		for (int ii = 0; ii < MaxConnections; ii++)		
			_ServiceThreads.push_back(Thread(ServiceThreadLauncher, (void*)this));			
		for (int ii = 0; ii < _ServiceThreads.size(); ii++)
			_ServiceThreads[ii].Start();
	}

	~Server()
	{
		_ShutdownEvent.Set();

		if (_pMainThread)
		{
			_pMainThread->Join();
			_pMainThread = nullptr;
		}

		for (auto ii = 0; ii < _ServiceThreads.size(); ii++) _ServiceThreads[ii].Join();
		_ServiceThreads.clear();
	}
};

#endif	// __XMLSocket_h__

//	End of XMLSocket.h

