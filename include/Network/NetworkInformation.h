/////////
//  NetworkInformation.h
//  Copyright (C) 2014 by Wiley Black
/////////

#ifndef __WBNetworkInformation_h__
#define __WBNetworkInformation_h__

/** Table of contents and references **/

namespace wb
{
	namespace net
	{
		namespace info
		{
			class NetworkInterface;
		}
	}
};

/** Dependencies **/

#include "../wbFoundation.h"
#include "Sockets.h"
#include "SocketExceptions.h"
#include "IP4Address.h"

#ifdef _WINDOWS
#include <iphlpapi.h>
#pragma comment(lib, "IPHLPAPI.lib")
#else
#include <sys/types.h>
#include <ifaddrs.h>
#include <net/if.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#endif

#include <vector>

/** Content **/

namespace wb
{
	namespace net
	{
		namespace info
		{
			using namespace wb::net::sockets;

			class NetworkInterface
			{								
				string			m_Name;
				unsigned int	m_Index;								// OS index of the network interface.
				IP4Address		m_IPAddress;
				IP4Address		m_Netmask;

				#ifdef _WINDOWS
				IP4Address		m_DefaultGateway;
				NetworkInterface(PIP_ADAPTER_INFO pInfo);
				#else
				NetworkInterface(ifaddrs* pInfo);
				#endif

			public:

				inline string GetName() { return m_Name; }
				inline IP4Address GetIPAddress() { return m_IPAddress; }
				inline IP4Address GetNetmask() { return m_Netmask; }
				inline IP4Address GetDefaultGateway();
				inline unsigned int GetIndex() { return m_Index; }
				
				static std::vector<NetworkInterface> GetAllNetworkInterfaces();
			};			

			/** Implementation **/

			#ifdef _WINDOWS

			inline NetworkInterface::NetworkInterface(PIP_ADAPTER_INFO pInfo)
			{
				/** Limitation: Windows allows multiple IP addresses per adapter, but we only
					grab the first one here. **/

				m_Name = pInfo->AdapterName;
				m_Index = pInfo->Index;
				m_IPAddress = IP4Address::Parse(pInfo->IpAddressList.IpAddress.String);
				m_Netmask = IP4Address::Parse(pInfo->IpAddressList.IpMask.String);
				m_DefaultGateway = IP4Address::Parse(pInfo->GatewayList.IpAddress.String);
			}

			inline /*static*/ vector<NetworkInterface> NetworkInterface::GetAllNetworkInterfaces()
			{
				vector<NetworkInterface> ret;

				ULONG ulOutBufLen = 0;
				DWORD dwRet = ::GetAdaptersInfo(nullptr, &ulOutBufLen);
				if (dwRet == NO_ERROR) return ret;			// No adapters found.
				if (dwRet != ERROR_BUFFER_OVERFLOW) Exception::ThrowFromWin32(dwRet);

				memory::Buffer AdapterInfoBuffer(ulOutBufLen);
				PIP_ADAPTER_INFO pAdapterInfo = (PIP_ADAPTER_INFO)AdapterInfoBuffer.At();

				dwRet = ::GetAdaptersInfo(pAdapterInfo, &ulOutBufLen);
				if (dwRet != NO_ERROR) Exception::ThrowFromWin32(dwRet);

				while (pAdapterInfo != nullptr)
				{
					ret.push_back(pAdapterInfo);
					pAdapterInfo = pAdapterInfo->Next;
				}

				return ret;
			}

			inline IP4Address NetworkInterface::GetDefaultGateway() { return m_DefaultGateway; }

			#else		/** Linux **/

			inline NetworkInterface::NetworkInterface(ifaddrs* pInfo)
			{
				m_Name = pInfo->ifa_name;
				m_Index = if_nametoindex(pInfo->ifa_name);
				if (pInfo->ifa_addr != nullptr) m_IPAddress = IP4Address(*pInfo->ifa_addr);
				if (pInfo->ifa_netmask != nullptr) m_Netmask = IP4Address(*pInfo->ifa_netmask);
			}

			inline /*static*/ vector<NetworkInterface> NetworkInterface::GetAllNetworkInterfaces()
			{
				ifaddrs *pInfoStart;
				if (getifaddrs(&pInfoStart) != 0) Exception::ThrowFromErrno(errno);
				vector<NetworkInterface> ret;
				try
				{
					ifaddrs *pInfo = pInfoStart;
					while (pInfo != nullptr)
					{
						if (pInfo->ifa_addr != nullptr && pInfo->ifa_addr->sa_family == AF_INET)
							ret.push_back(NetworkInterface(pInfo));
						pInfo = pInfo->ifa_next;
					}
				}
				catch (std::exception&)
				{
					freeifaddrs(pInfoStart);
					throw;
				}
				freeifaddrs(pInfoStart);
				return ret;
			}

			inline IP4Address NetworkInterface::GetDefaultGateway()
			{
				static unsigned int SeqNumber = 0;
				const int BufferSize = 65536;

				SeqNumber ++;
				UInt32 PID = getpid();

				/** Open a special socket that allows us to retrieve configuration information from the kernel **/
				SOCKET sNetlink = socket(PF_NETLINK, SOCK_DGRAM, NETLINK_ROUTE);
				if (sNetlink == INVALID_SOCKET) 				
					throw SocketException("Unable to establish netlink socket with kernel for configuration query.");

				/** Bind the socket to particular settings **/
				sockaddr_nl SAddr;
				ZeroMemory (&SAddr, sizeof(SAddr));
				SAddr.nl_family = AF_NETLINK;
				SAddr.nl_groups = RTMGRP_LINK | RTMGRP_IPV4_IFADDR;
				if (bind(sNetlink, (sockaddr*)&SAddr, sizeof(SAddr)) == SOCKET_ERROR)
					throw SocketException("Error binding to netlink/route socket for configuration query.", GetLastSocketError());
				
				{
					/** Setup request message **/				
					char Request[BufferSize];
					ZeroMemory (Request, sizeof(Request));
					nlmsghdr* pHeader = (nlmsghdr*)Request;
					pHeader->nlmsg_len = NLMSG_LENGTH(sizeof(rtmsg));
					assert (sizeof(Request) >= sizeof(nlmsghdr) + sizeof(rtmsg));
					pHeader->nlmsg_type = RTM_GETROUTE;
					pHeader->nlmsg_flags = NLM_F_DUMP | NLM_F_REQUEST;				// The message is a request for dump.
					pHeader->nlmsg_seq = SeqNumber;
					pHeader->nlmsg_pid = PID;										// Sender PID
				
					/* Send the request */
					if (send(sNetlink, pHeader, pHeader->nlmsg_len, 0) == SOCKET_ERROR)
						throw SocketException("Error making netlink/route configuration request.", GetLastSocketError());
				}
				
				// We will search for a default gateway specifically configured for the requested ethernet adapter,
				// however if we do not find one then we will return the default gateway from any ethernet adapter
				// in the system.
				bool AnyGatewayFound = false;
				IP4Address SystemGateway;

				{
					/** Retrieve response from the kernel **/
					char Response[BufferSize];
					nlmsghdr *pHeader;
					size_t readLen = 0, msgLen = 0;

					for(;;)
					{
						if (msgLen >= (size_t)BufferSize)
							throw SocketException("Extremely large routing table retrieval not supported.");

						readLen = recv(sNetlink, Response + msgLen, BufferSize - msgLen, 0);
						if (readLen == (size_t)SOCKET_ERROR) 
							throw SocketException("Error receiving netlink/route configuration.", GetLastSocketError());						

						pHeader = (nlmsghdr *)(Response + msgLen);

						/* Check if the header is valid */
						if ((NLMSG_OK(pHeader, readLen) == 0) || (pHeader->nlmsg_type == NLMSG_ERROR))
							throw SocketException("Error in received netlink/route configuration.");						

						if ((pHeader->nlmsg_seq != SeqNumber) || (pHeader->nlmsg_pid != PID)) continue;						

						/* Check if it's the last message */
						if (pHeader->nlmsg_type == NLMSG_DONE) break;
						else msgLen += readLen;

						/* Check if it's a multi part message */
						if((pHeader->nlmsg_flags & NLM_F_MULTI) == 0) break;
					} 

					/** Parse response from kernel **/					

					for(pHeader = (nlmsghdr*)Response; NLMSG_OK(pHeader,msgLen); pHeader = NLMSG_NEXT(pHeader,msgLen))
					{
						rtmsg* pRouteMessage = (rtmsg*)NLMSG_DATA(pHeader);
						rtattr* pRouteAttr;

						/* If the route is not for AF_INET or does not belong to main routing table then return. */
						if ((pRouteMessage->rtm_family != AF_INET) || (pRouteMessage->rtm_table != RT_TABLE_MAIN)) continue;
						
						IP4Address Gateway;
						IP4Address DstAddr;
						IP4Address SrcAddr;						

						pRouteAttr = (rtattr*)RTM_RTA(pRouteMessage);
						int RouteLength = RTM_PAYLOAD(pHeader);
						bool IsMatchToInterface = false;
						for(; RTA_OK(pRouteAttr,RouteLength); pRouteAttr = RTA_NEXT(pRouteAttr,RouteLength))
						{							
							switch(pRouteAttr->rta_type) {
							case RTA_OIF:
								{
									unsigned int InterfaceIndex = *(int *)RTA_DATA(pRouteAttr);
									IsMatchToInterface = (InterfaceIndex == m_Index);
									break;
								}
							case RTA_GATEWAY:
								Gateway = IP4Address(htonl(*(u_int *)RTA_DATA(pRouteAttr)));
								break;
							case RTA_PREFSRC:
								SrcAddr = IP4Address(htonl(*(u_int *)RTA_DATA(pRouteAttr)));
								break;
							case RTA_DST:
								DstAddr = IP4Address(htonl(*(u_int *)RTA_DATA(pRouteAttr)));
								break;
							}
						}

						if (IsMatchToInterface && DstAddr.GetIP() == 0) return Gateway;
						if (DstAddr.GetIP() == 0) { AnyGatewayFound = true; SystemGateway = Gateway; }
					}
				}
					
				if (AnyGatewayFound) return SystemGateway;

				throw NetworkException("Could not identify network gateway address for interface.");
			}
			#endif
		}
	}
}

#endif	// __NetworkInformation_h__

//  End of NetworkInformation.h

