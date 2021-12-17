/*	Exceptions.h
	Copyright (C) 2014 by Wiley Black (TheWiley@gmail.com)

	Example Usage:

		try
		{
			// Execute some code...
			if (ThingsGoBadly) throw NetworkException("An example of a specific exception.");
			// Call some other code that might throw a std::exception...
		}
		catch (wb::NetworkException& ex)
		{
			printf("A network error occurred:\n    %s\n", ex.GetMessageW().c_str());
			printf("Press any key to exit.\n");
			_getch();
		}
		catch (std::exception& ex)
		{
			printf("Exception occurred:\n    %s\n", ex.what());
			printf("Press any key to exit.\n");
			_getch();
		}
*/

#ifndef __wbStdException_h__
#define __wbStdException_h__

#ifndef EmulateSTL
#include <stdexcept>
#else

namespace std
{
	// Although we will not be throwing any std::exception's from our code, defining it
	// allows us to catch std::exception's whether using STL or not.
	class exception
	{   // base of all library exceptions
	public:
		exception() { msg = ""; }
		exception(const char* const& _msg) { msg = _msg; }
		exception(const exception& cp) { msg = cp.msg; }
		exception& operator=(const exception& cp) { msg = cp.msg; return *this; }
		virtual ~exception() { }
		virtual const char* what() const throw() { return msg; }

	private:
		const char* msg;
	};
}

#endif		// EmulateSTL

#endif	// __StdExceptions_h__

//	End of StdExceptions.h

