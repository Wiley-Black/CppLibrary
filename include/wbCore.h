/** Provides the "Core" component of Wiley Black's C++ Library **/

#ifndef __wbCore__
#define __wbCore__

#include "wbFoundation.h"

#include "Network/Sockets.h"
#include "Network/NetworkInformation.h"

#include "Foundation/STL/Collections/Stack.h"
#include "Foundation/STL/Collections/UnorderedMap.h"
#include "Foundation/STL/Collections/Vector.h"

#include "Text/StringBuilder.h"

#include "IO/FileStream.h"
#include "IO/MemoryStream.h"
#include "IO/Console.h"
#include "IO/File.h"
#include "IO/FileInfo.h"
#include "IO/Path.h"
#include "IO/StreamWriter.h"
#include "IO/BinaryWriter.h"
#include "IO/BinaryReader.h"

#include "Math/Random.h"

#include "Foundation/STL/Memory.h"
#include "Memory Management/Buffer.h"
#include "Memory Management/FixedRingBuffer.h"
#include "Memory Management/FlexRingBuffer.h"
#include "Memory Management/FlexRingBufferEx.h"
#include "Memory Management/HistoryRingBuffer.h"
#include "Memory Management/MemoryInfo.h"

#include "Parsing/BaseTypeParsing.h"
#include "Parsing/Xml/Xml.h"
#include "Parsing/Xml/XmlParser.h"
#include "Parsing/Json/Json.h"
#include "Parsing/Json/JsonParser.h"
#include "Parsing/Yaml/Yaml.h"
#include "Parsing/Yaml/YamlParser.h"

#include "System/Diagnostics.h"
#include "System/Environment.h"
#include "System/Guid.h"
#include "System/Monitor.h"
#include "System/Process.h"
#include "System/Stopwatch.h"
#include "System/Threading.h"
#include "System/Registry.h"

#include "Processing/CRC.h"

#endif // __wbCore__
