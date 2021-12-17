/** Provides the "Foundation" component of Wiley Black's C++ Library.
* 
*	The "Foundation" component is the absolute minimum useful portion of the library.
*	All other components of WB's library require the Foundation.
*/

#ifndef __wbFoundation__
#define __wbFoundation__

#include "wbConfiguration.h"
#include "Foundation/Common.h"

/** Socket Dependencies - must come before certain Windows header files to preclude Sockets v1 from being included **/
#ifdef __wbCore__
#include "Network/SocketDependencies.h"
#endif

#include "Foundation/STL/cstddef.h"				// For nullptr and nullptr_t, if emulating.
#include "Foundation/STL/utility.h"				// For remove_reference and miscellaneous helpers.
#include "Foundation/Exceptions.h"
#include "Text/Encoding.h"						// Referenced by Foundation/Exceptions.h.

#include "Foundation/STL/Text/String.h"
#include "Text/StringHelpers.h"					// Referenced by Foundation/Exceptions.h.
#include "Text/StringComparison.h"
#include "Text/StringConversion.h"

#include "IO/Streams.h"

#include "Platforms/Language.h"
#include "Platforms/Platforms.h"

#endif // __wbFoundation__
