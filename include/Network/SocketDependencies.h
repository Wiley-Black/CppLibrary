#ifndef __wbSocketDependencies_h__
#define __wbSocketDependencies_h__

#if defined(_WIN32) || defined(_WIN64)
#if defined(__AFXWIN_H__) && !defined(_AFX_NO_SOCKET_SUPPORT)
#	error Common.h must be included before AfxWin.h.
#endif

#ifndef _AFX_NO_SOCKET_SUPPORT
#define _AFX_NO_SOCKET_SUPPORT
#endif

#if defined(WIN32)
#	include <SDKDDKVer.h>
#	if !defined(_WIN32_WINNT) || (_WIN32_WINNT < 0x0400)
#		error Windows Sockets V2.0 is required, which requires WINNT 4.0, so define _WIN32_WINNT=0x0400.
#	endif
#endif

#include <WinSock2.h>
#pragma comment( lib, "WS2_32.LIB" )				// Sounds like this is also the name of the 64-bit version.
#define	SOCKETS_VERSION		MAKEWORD( /*major*/ 2, /*minor*/ 2 )
#endif

#endif	// __wbSocketDependencies_h__
