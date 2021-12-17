/////////
//	Console.h
//	Copyright (C) 2014 by Wiley Black
////

#ifndef __WBConsole_h__
#define __WBConsole_h__

#include "wbFoundation.h"

#if defined(_WINDOWS)
#include <conio.h>
#else
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>
#include <fcntl.h>
#include <termios.h>
#endif

namespace wb
{
	namespace io
	{
		class Console
		{
			#ifndef _WINDOWS
			struct termios oldSettings, newSettings;
			#endif

		public:
			Console()
			{
				#ifndef _WINDOWS				
				tcgetattr( fileno( stdin ), &oldSettings );
				newSettings = oldSettings;
				newSettings.c_lflag &= (~ICANON & ~ECHO);
				tcsetattr( fileno( stdin ), TCSANOW, &newSettings );
				#endif
			}

			~Console()
			{
				#ifndef _WINDOWS
				tcsetattr( fileno( stdin ), TCSANOW, &oldSettings );
				#endif
			}

			char Read()
			{
				#ifdef _WINDOWS
				return getchar();
				#else
				return getchar();
				#endif
			}

			bool IsKeyAvailable()
			{
				#ifdef _WINDOWS

				return (_kbhit() != 0);

				#else

				fd_set set;
				struct timeval tv;
				tv.tv_sec = 0; tv.tv_usec = 0;

				FD_ZERO( &set );
				FD_SET( fileno( stdin ), &set );

				int result = select( fileno( stdin )+1, &set, NULL, NULL, &tv );
				if (result < 0) Exception::ThrowFromErrno(errno);
				return (result > 0);

				#endif
			}
		};		
	}
}

#endif	// __WBConsole_h__

//	End of Console.h

