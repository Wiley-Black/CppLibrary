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

#ifndef __WBExceptions_h__
#define __WBExceptions_h__

#include "../Foundation/STL/StdException.h"
#include "../Platforms/Platforms.h"
#include "../Foundation/STL/Text/String.h"

#ifdef _WINDOWS
#include <comutil.h>

// In Microsoft Visual C++ 2019 using the NVIDIA nvcc compiler, including <comdef.h> leads to the following error:
// C:\Program Files(x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.28.29910\include\comdef.h(494) : error: invalid nontype template argument of type "const _GUID *"
// And this error appears even if comdef.h. is the first include file in a .cu file.  At present, it seems that comdef.h is only needed for code that is commented out below,
// and so we can just avoid it for the time being.
//#include <comdef.h>

#include <WbemCli.h>
#endif

#include <errno.h>

#undef GetMessage			// Avoid conflict with MSVC macro.

namespace wb
{
	class Exception : public std::exception
	{
	protected:
		// Although the Microsoft implementation of std::exception includes a message,
		// this allows portability.
		std::string Message;

	public:
		Exception() { }
		Exception(const char* const& message) { Message = message; }
		Exception(const std::string& message) { Message = message; }
		Exception(const Exception& right) { Message = right.Message; }
		Exception(Exception&& from) { Message = std::move(from.Message); }
		Exception& operator=(const Exception& right) {
			std::exception::operator=(right);
			Message = right.Message;
			return *this;
		}
		~Exception() throw() { }
		virtual std::string GetMessage() const { return Message; }
		inline std::string GetMessageA() const { return Message; }		// In case the MSVC macros interfere.
		inline std::string GetMessageW() const { return Message; }		// In case the MSVC macros interfere.
		const char* what() const throw() override { return Message.c_str(); }

		static void ThrowFromErrno(int FromErrno);
#if defined(_WINDOWS)
		static void ThrowFromWin32(UInt32 LastError);
		static void ThrowFromHRESULT(HRESULT LastError);
		//template<class T> static void ThrowFromHRESULT(HRESULT LastError, T* pObject);	
#endif
	};

#	define DeclareGenericException(ExceptionClass,DefaultMessage)			\
	class ExceptionClass : public Exception {				\
		public:		\
			ExceptionClass() : Exception(DefaultMessage) { }			\
			ExceptionClass(const char * const &message) : Exception(message) { }	\
			ExceptionClass(const std::string& message) : Exception(message) { }	\
			ExceptionClass(const ExceptionClass &right) : Exception(right) { }	\
			ExceptionClass(ExceptionClass&& from) : Exception(from) { }	\
			ExceptionClass& operator=(const ExceptionClass &right) { Exception::operator=(right); return *this; }		\
	}

	DeclareGenericException(ArgumentException, S("An invalid argument was supplied."));
	DeclareGenericException(ArgumentNullException, S("Argument provided was null."));
	DeclareGenericException(ArgumentOutOfRangeException, S("Out of range."));
	DeclareGenericException(DirectoryNotFoundException, S("Directory not found."));
	DeclareGenericException(EndOfStreamException, S("Unexpected end of stream."));
	DeclareGenericException(FileNotFoundException, S("File not found."));
	DeclareGenericException(IndexOutOfRangeException, S("Index outside of valid range."));
	DeclareGenericException(IOException, S("An I/O error occurred."));
	DeclareGenericException(UnauthorizedAccessException, S("Unauthorized access."));
	DeclareGenericException(NotSupportedException, S("Operation not supported."));
	DeclareGenericException(NotImplementedException, S("Operation not implemented."));
	DeclareGenericException(OutOfMemoryException, S("Out of memory."));
	DeclareGenericException(FormatException, S("Invalid format."));
	DeclareGenericException(TimeoutException, S("A timeout has occurred."));
	DeclareGenericException(NetworkException, S("Network communication failure."));

#ifdef _WINDOWS
	class COMException : public Exception {
	public:
		HRESULT Code;

		COMException() : Exception("An unrecognized COM error has occurred.") { Code = 0; }
		COMException(const COMException& right) : Exception(right) { Code = right.Code; }
		COMException(COMException&& from) : Exception(from) { Code = from.Code; }
		COMException& operator=(const COMException& right) { Exception::operator=(right); Code = right.Code; return *this; }
		COMException(HRESULT ErrorCode) : Exception()
		{
			char tmp[256];
			sprintf_s(tmp, sizeof(tmp), "COM Error 0x%08X has occurred.", ErrorCode);
			Message = tmp;
			Code = ErrorCode;
		}
		COMException(const std::string& message, HRESULT ErrorCode) : Exception(message) { Code = ErrorCode; }
	};
#endif
}

/** Late Dependencies **/
#include "Text/StringHelpers.h"
#include "Text/Encoding.h"

namespace wb
{
	/** Implementations **/

	inline /*static*/ void Exception::ThrowFromErrno(int FromErrno)
	{
		switch (FromErrno)
		{
		case EADDRNOTAVAIL: throw Exception("The given address is not available.");
		case EADDRINUSE: throw Exception("The given address is already in use.");
		case ENOTSOCK: throw Exception("Given descriptor is for a file, not a socket.");
		case ELOOP: throw Exception("Too many symbolic links were encountered in resolving addr.");
		case EPERM: throw Exception(S("Operation not permitted"));
		case ENOENT: throw FileNotFoundException();
		case ESRCH: throw Exception(S("No such process"));
		case EINTR: throw Exception(S("Interrupted function"));
		case EIO: throw IOException();
		case ENXIO: throw Exception(S("No such device or address"));
		case E2BIG: throw Exception(S("Argument list too long"));
		case ENOEXEC: throw Exception(S("Executable format error"));
		case EBADF: throw Exception(S("Bad file number/descriptor"));
		case ECHILD: throw Exception(S("No spawned processes"));
		case EAGAIN: throw Exception(S("No more processes or not enough memory or maximum nesting level reached"));
		case ENOMEM: throw OutOfMemoryException(S("Not enough memory"));
		case EACCES: throw UnauthorizedAccessException(S("Permission denied"));
		case EFAULT: throw Exception(S("Bad address"));
		case EBUSY: throw Exception(S("Device or resource busy"));
		case EEXIST: throw Exception(S("File exists"));
		case EXDEV: throw Exception(S("Cross-device link"));
		case ENODEV: throw Exception(S("No such device"));
		case ENOTDIR: throw Exception(S("Not a directory"));
		case EISDIR: throw Exception(S("Is a directory"));
		case EINVAL: throw Exception(S("Invalid argument or operation"));
		case ENFILE: throw Exception(S("Too many files open in system"));
		case EMFILE: throw Exception(S("Too many open files"));
		case ENOTTY: throw Exception(S("Inappropriate I/O control operation"));
		case EFBIG: throw Exception(S("File too large"));
		case ENOSPC: throw Exception(S("No space left on device"));
		case ESPIPE: throw Exception(S("Invalid seek"));
		case EROFS: throw Exception(S("Read-only file system"));
		case EMLINK: throw Exception(S("Too many links"));
		case EPIPE: throw Exception(S("Broken pipe"));
		case EDOM: throw Exception(S("Math argument"));
		case ERANGE: throw Exception(S("Result too large"));
		case EDEADLK: throw Exception(S("Resource deadlock would occur"));
		case ENAMETOOLONG: throw Exception(S("Filename too long"));
		case ENOLCK: throw Exception(S("No locks available"));
		case ENOSYS: throw Exception(S("Function not supported"));
		case ENOTEMPTY: throw Exception(S("Directory not empty"));
		case EILSEQ: throw Exception(S("Illegal byte sequence"));
#ifdef _MSC_VER
		case STRUNCATE: throw Exception(S("String was truncated"));
#endif
		default: throw Exception("Unrecognized system error (" + std::to_string(FromErrno) + ")");
		}
	}

#if defined(_WINDOWS)
	inline /*static*/ void Exception::ThrowFromWin32(UInt32 LastError)
	{
		TCHAR* pszBuffer;
		if (::FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, LastError, 0, (LPTSTR)&pszBuffer, sizeof(TCHAR*), nullptr) == 0)
		{
			char ret[20]; sprintf_s(ret, sizeof(ret), "%X", LastError);
			throw Exception(("System error 0x" + std::string(ret) + ", unable to retrieve detailed error information.").c_str());
		}
		osstring os_msg = pszBuffer;
		::LocalFree(pszBuffer);
		std::string msg = wb::to_string(os_msg);

		switch (LastError)
		{
		case ERROR_FILE_NOT_FOUND: throw FileNotFoundException(msg.c_str());
		case ERROR_PATH_NOT_FOUND: throw DirectoryNotFoundException(msg.c_str());
		case ERROR_ACCESS_DENIED: throw UnauthorizedAccessException(msg.c_str());
		case ERROR_NOT_ENOUGH_MEMORY: throw OutOfMemoryException(msg.c_str());
		case ERROR_BAD_FORMAT: throw FormatException(msg.c_str());
		case ERROR_INVALID_ACCESS: throw UnauthorizedAccessException(msg.c_str());
		case ERROR_INVALID_PARAMETER: throw ArgumentException(msg.c_str());
		case ERROR_NOT_SUPPORTED: throw NotSupportedException(msg.c_str());
		case ERROR_OUTOFMEMORY: throw OutOfMemoryException(msg.c_str());
		case ERROR_WRITE_FAULT: throw IOException(msg.c_str());
		case ERROR_READ_FAULT: throw IOException(msg.c_str());
		case ERROR_HANDLE_EOF: throw EndOfStreamException(msg.c_str());
		case ERROR_HANDLE_DISK_FULL: throw IOException(msg.c_str());
		default: throw Exception(msg.c_str());
		}
	}

	inline /*static*/ void Exception::ThrowFromHRESULT(HRESULT LastError)
	{
		switch (LastError)
		{
			/** WMI Errors **/
		case WBEM_E_FAILED: throw Exception("Call failed.");
		case WBEM_E_NOT_FOUND: throw Exception("Object cannot be found.");
		case WBEM_E_ACCESS_DENIED: throw Exception("Current user does not have permission to perform the action.");
		case WBEM_E_PROVIDER_FAILURE: throw Exception("Provider has failed at some time other than during initialization.");
		case WBEM_E_TYPE_MISMATCH: throw Exception("Type mismatch occurred.");
		case WBEM_E_OUT_OF_MEMORY: throw Exception("Not enough memory for the operation.");
		case WBEM_E_INVALID_CONTEXT: throw Exception("The IWbemContext object is not valid.");
		case WBEM_E_INVALID_PARAMETER: throw Exception("One of the parameters to the call is not correct.");
		case WBEM_E_NOT_AVAILABLE: throw Exception("Resource, typically a remote server, is not currently available.");
		case WBEM_E_CRITICAL_ERROR: throw Exception("Internal, critical, and unexpected error occurred. Report the error to Microsoft Technical Support.");
		case WBEM_E_INVALID_STREAM: throw Exception("One or more network packets were corrupted during a remote session.");
		case WBEM_E_NOT_SUPPORTED: throw Exception("Feature or operation is not supported.");
		case WBEM_E_INVALID_SUPERCLASS: throw Exception("Parent class specified is not valid.");
		case WBEM_E_INVALID_NAMESPACE: throw Exception("Namespace specified cannot be found.");
		case WBEM_E_INVALID_OBJECT: throw Exception("Specified instance is not valid.");
		case WBEM_E_INVALID_CLASS: throw Exception("Specified class is not valid.");
		case WBEM_E_PROVIDER_NOT_FOUND: throw Exception("Provider referenced in the schema does not have a corresponding registration.");
		case WBEM_E_INVALID_PROVIDER_REGISTRATION: throw Exception("Provider referenced in the schema has an incorrect or incomplete registration.");
		case WBEM_E_PROVIDER_LOAD_FAILURE: throw Exception("COM cannot locate a provider referenced in the schema.");
		case WBEM_E_INITIALIZATION_FAILURE: throw Exception("Component, such as a provider, failed to initialize for internal reasons.");
		case WBEM_E_TRANSPORT_FAILURE: throw Exception("Networking error that prevents normal operation has occurred.");
		case WBEM_E_INVALID_OPERATION: throw Exception("Requested operation is not valid. This error usually applies to invalid attempts to delete classes or properties.");
		case WBEM_E_INVALID_QUERY: throw Exception("Query was not syntactically valid.");
		case WBEM_E_INVALID_QUERY_TYPE: throw Exception("Requested query language is not supported.");
		case WBEM_E_ALREADY_EXISTS: throw Exception("In a put operation, the wbemChangeFlagCreateOnly flag was specified, but the instance already exists.");
		case WBEM_E_OVERRIDE_NOT_ALLOWED: throw Exception("Not possible to perform the add operation on this qualifier because the owning object does not permit overrides.");
		case WBEM_E_PROPAGATED_QUALIFIER: throw Exception("User attempted to delete a qualifier that was not owned. The qualifier was inherited from a parent class.");
		case WBEM_E_PROPAGATED_PROPERTY: throw Exception("User attempted to delete a property that was not owned. The property was inherited from a parent class.");
		case WBEM_E_UNEXPECTED: throw Exception("Client made an unexpected and illegal sequence of calls, such as calling EndEnumeration before calling BeginEnumeration.");
		case WBEM_E_ILLEGAL_OPERATION: throw Exception("User requested an illegal operation, such as spawning a class from an instance.");
		case WBEM_E_CANNOT_BE_KEY: throw Exception("Illegal attempt to specify a key qualifier on a property that cannot be a key. The keys are specified in the class definition for an object and cannot be altered on a per-instance basis.");
		case WBEM_E_INCOMPLETE_CLASS: throw Exception("Current object is not a valid class definition. Either it is incomplete or it has not been registered with WMI using SWbemObject.Put_.");
		case WBEM_E_INVALID_SYNTAX: throw Exception("Query is syntactically not valid.");
		case WBEM_E_NONDECORATED_OBJECT: throw Exception("Reserved for future use.");
		case WBEM_E_READ_ONLY: throw Exception("An attempt was made to modify a read-only property.");
		case WBEM_E_PROVIDER_NOT_CAPABLE: throw Exception("Provider cannot perform the requested operation. This can include a query that is too complex, retrieving an instance, creating or updating a class, deleting a class, or enumerating a class.");
		case WBEM_E_CLASS_HAS_CHILDREN: throw Exception("Attempt was made to make a change that invalidates a subclass.");
		case WBEM_E_CLASS_HAS_INSTANCES: throw Exception("Attempt was made to delete or modify a class that has instances.");
		case WBEM_E_QUERY_NOT_IMPLEMENTED: throw Exception("Reserved for future use.");
		case WBEM_E_ILLEGAL_NULL: throw Exception("Value of Nothing/NULL was specified for a property that must have a value, such as one that is marked by a Key, Indexed, or Not_Null qualifier.");
		case WBEM_E_INVALID_QUALIFIER_TYPE: throw Exception("Variant value for a qualifier was provided that is not a legal qualifier type.");
		case WBEM_E_INVALID_PROPERTY_TYPE: throw Exception("CIM type specified for a property is not valid.");
		case WBEM_E_VALUE_OUT_OF_RANGE: throw Exception("Request was made with an out-of-range value or it is incompatible with the type.");
		case WBEM_E_CANNOT_BE_SINGLETON: throw Exception("Illegal attempt was made to make a class singleton, such as when the class is derived from a non-singleton class.");
		case WBEM_E_INVALID_CIM_TYPE: throw Exception("CIM type specified is not valid.");
		case WBEM_E_INVALID_METHOD: throw Exception("Requested method is not available.");
		case WBEM_E_INVALID_METHOD_PARAMETERS: throw Exception("Parameters provided for the method are not valid.");
		case WBEM_E_SYSTEM_PROPERTY: throw Exception("There was an attempt to get qualifiers on a system property.");
		case WBEM_E_INVALID_PROPERTY: throw Exception("Property type is not recognized.");
		case WBEM_E_CALL_CANCELLED: throw Exception("Asynchronous process has been canceled internally or by the user. Note that due to the timing and nature of the asynchronous operation, the operation may not have been truly canceled.");
		case WBEM_E_SHUTTING_DOWN: throw Exception("User has requested an operation while WMI is in the process of shutting down.");
		case WBEM_E_PROPAGATED_METHOD: throw Exception("Attempt was made to reuse an existing method name from a parent class and the signatures do not match.");
		case WBEM_E_UNSUPPORTED_PARAMETER: throw Exception("One or more parameter values, such as a query text, is too complex or unsupported. WMI is therefore requested to retry the operation with simpler parameters.");
		case WBEM_E_MISSING_PARAMETER_ID: throw Exception("Parameter was missing from the method call.");
		case WBEM_E_INVALID_PARAMETER_ID: throw Exception("Method parameter has an ID qualifier that is not valid.");
		case WBEM_E_NONCONSECUTIVE_PARAMETER_IDS: throw Exception("One or more of the method parameters have ID qualifiers that are out of sequence.");
		case WBEM_E_PARAMETER_ID_ON_RETVAL: throw Exception("Return value for a method has an ID qualifier.");
		case WBEM_E_INVALID_OBJECT_PATH: throw Exception("Specified object path was not valid.");
		case WBEM_E_OUT_OF_DISK_SPACE: throw Exception("Disk is out of space or the 4 GB limit on WMI repository (CIM repository) size is reached.");
		case WBEM_E_BUFFER_TOO_SMALL: throw Exception("Supplied buffer was too small to hold all of the objects in the enumerator or to read a std::string property.");
		case WBEM_E_UNSUPPORTED_PUT_EXTENSION: throw Exception("Provider does not support the requested put operation.");
		case WBEM_E_UNKNOWN_OBJECT_TYPE: throw Exception("Object with an incorrect type or version was encountered during marshaling.");
		case WBEM_E_UNKNOWN_PACKET_TYPE: throw Exception("Packet with an incorrect type or version was encountered during marshaling.");
		case WBEM_E_MARSHAL_VERSION_MISMATCH: throw Exception("Packet has an unsupported version.");
		case WBEM_E_MARSHAL_INVALID_SIGNATURE: throw Exception("Packet appears to be corrupt.");
		case WBEM_E_INVALID_QUALIFIER: throw Exception("Attempt was made to mismatch qualifiers, such as putting [key] on an object instead of a property.");
		case WBEM_E_INVALID_DUPLICATE_PARAMETER: throw Exception("Duplicate parameter was declared in a CIM method.");
		case WBEM_E_TOO_MUCH_DATA: throw Exception("Reserved for future use.");
		case WBEM_E_SERVER_TOO_BUSY: throw Exception("Call to IWbemObjectSink::Indicate has failed. The provider can refire the event.");
		case WBEM_E_INVALID_FLAVOR: throw Exception("Specified qualifier flavor was not valid.");
		case WBEM_E_CIRCULAR_REFERENCE: throw Exception("Attempt was made to create a reference that is circular (for example, deriving a class from itself).");
		case WBEM_E_UNSUPPORTED_CLASS_UPDATE: throw Exception("Specified class is not supported.");
		case WBEM_E_CANNOT_CHANGE_KEY_INHERITANCE: throw Exception("Attempt was made to change a key when instances or subclasses are already using the key.");
		case WBEM_E_CANNOT_CHANGE_INDEX_INHERITANCE: throw Exception("An attempt was made to change an index when instances or subclasses are already using the index.");
		case WBEM_E_TOO_MANY_PROPERTIES: throw Exception("Attempt was made to create more properties than the current version of the class supports.");
		case WBEM_E_UPDATE_TYPE_MISMATCH: throw Exception("Property was redefined with a conflicting type in a derived class.");
		case WBEM_E_UPDATE_OVERRIDE_NOT_ALLOWED: throw Exception("Attempt was made in a derived class to override a qualifier that cannot be overridden.");
		case WBEM_E_UPDATE_PROPAGATED_METHOD: throw Exception("Method was re-declared with a conflicting signature in a derived class.");
		case WBEM_E_METHOD_NOT_IMPLEMENTED: throw Exception("Attempt was made to execute a method not marked with [implemented] in any relevant class.");
		case WBEM_E_METHOD_DISABLED: throw Exception("Attempt was made to execute a method marked with [disabled].");
		case WBEM_E_REFRESHER_BUSY: throw Exception("Refresher is busy with another operation.");
		case WBEM_E_UNPARSABLE_QUERY: throw Exception("Filtering query is syntactically not valid.");
		case WBEM_E_NOT_EVENT_CLASS: throw Exception("The FROM clause of a filtering query references a class that is not an event class (not derived from __Event).");
		case WBEM_E_MISSING_GROUP_WITHIN: throw Exception("A GROUP BY clause was used without the corresponding GROUP WITHIN clause.");
		case WBEM_E_MISSING_AGGREGATION_LIST: throw Exception("A GROUP BY clause was used. Aggregation on all properties is not supported.");
		case WBEM_E_PROPERTY_NOT_AN_OBJECT: throw Exception("Dot notation was used on a property that is not an embedded object.");
		case WBEM_E_AGGREGATING_BY_OBJECT: throw Exception("A GROUP BY clause references a property that is an embedded object without using dot notation.");
		case WBEM_E_UNINTERPRETABLE_PROVIDER_QUERY: throw Exception("Event provider registration query (__EventProviderRegistration) did not specify the classes for which events were provided.");
		case WBEM_E_BACKUP_RESTORE_WINMGMT_RUNNING: throw Exception("Request was made to back up or restore the repository while it was in use by WinMgmt.exe, or by the SVCHOST process that contains the WMI service.");
		case WBEM_E_QUEUE_OVERFLOW: throw Exception("Asynchronous delivery queue overflowed from the event consumer being too slow.");
		case WBEM_E_PRIVILEGE_NOT_HELD: throw Exception("Operation failed because the client did not have the necessary security privilege.");
		case WBEM_E_INVALID_OPERATOR: throw Exception("Operator is not valid for this property type.");
		case WBEM_E_LOCAL_CREDENTIALS: throw Exception("User specified a username/password/authority on a local connection. The user must use a blank username/password and rely on default security.");
		case WBEM_E_CANNOT_BE_ABSTRACT: throw Exception("Class was made abstract when its parent class is not abstract.");
		case WBEM_E_AMENDED_OBJECT: throw Exception("Amended object was written without the WBEM_FLAG_USE_AMENDED_QUALIFIERS flag being specified.");
		case WBEM_E_CLIENT_TOO_SLOW: throw Exception("Client did not retrieve objects quickly enough from an enumeration. This constant is returned when a client creates an enumeration object, but does not retrieve objects from the enumerator in a timely fashion, causing the enumerator's object caches to back up.");
		case WBEM_E_NULL_SECURITY_DESCRIPTOR: throw Exception("Null security descriptor was used.");
		case WBEM_E_TIMED_OUT: throw Exception("Operation timed out.");
		case WBEM_E_INVALID_ASSOCIATION: throw Exception("Association is not valid.");
		case WBEM_E_AMBIGUOUS_OPERATION: throw Exception("Operation was ambiguous.");
		case WBEM_E_QUOTA_VIOLATION: throw Exception("WMI is taking up too much memory. This can be caused by low memory availability or excessive memory consumption by WMI.");
		case 0x8004106D: throw Exception("Operation resulted in a transaction conflict.");	// WBEM_E_TRANSACTION_CONFLICT
		case 0x8004106E: throw Exception("Transaction forced a rollback.");	// WBEM_E_FORCED_ROLLBACK
		case WBEM_E_UNSUPPORTED_LOCALE: throw Exception("Locale used in the call is not supported.");
		case WBEM_E_HANDLE_OUT_OF_DATE: throw Exception("Object handle is out-of-date.");
		case WBEM_E_CONNECTION_FAILED: throw Exception("Connection to the SQL database failed.");
		case WBEM_E_INVALID_HANDLE_REQUEST: throw Exception("Handle request was not valid.");
		case WBEM_E_PROPERTY_NAME_TOO_WIDE: throw Exception("Property name contains more than 255 characters.");
		case WBEM_E_CLASS_NAME_TOO_WIDE: throw Exception("Class name contains more than 255 characters.");
		case WBEM_E_METHOD_NAME_TOO_WIDE: throw Exception("Method name contains more than 255 characters.");
		case WBEM_E_QUALIFIER_NAME_TOO_WIDE: throw Exception("Qualifier name contains more than 255 characters.");
		case WBEM_E_RERUN_COMMAND: throw Exception("The SQL command must be rerun because there is a deadlock in SQL. This can be returned only when data is being stored in an SQL database.");
		case WBEM_E_DATABASE_VER_MISMATCH: throw Exception("The database version does not match the version that the repository driver processes.");
		case WBEM_E_VETO_DELETE: throw Exception("WMI cannot execute the delete operation because the provider does not allow it.");
		case WBEM_E_VETO_PUT: throw Exception("WMI cannot execute the put operation because the provider does not allow it.");
		case WBEM_E_INVALID_LOCALE: throw Exception("Specified locale identifier was not valid for the operation.");
		case WBEM_E_PROVIDER_SUSPENDED: throw Exception("Provider is suspended.");
		case WBEM_E_SYNCHRONIZATION_REQUIRED: throw Exception("Object must be written to the WMI repository and retrieved again before the requested operation can succeed. This constant is returned when an object must be committed and retrieved to see the property value.");
		case WBEM_E_NO_SCHEMA: throw Exception("Operation cannot be completed; no schema is available.");
		case WBEM_E_PROVIDER_ALREADY_REGISTERED: throw Exception("Provider cannot be registered because it is already registered.");
		case WBEM_E_PROVIDER_NOT_REGISTERED: throw Exception("Provider was not registered.");
		case WBEM_E_FATAL_TRANSPORT_ERROR: throw Exception("A fatal transport error occurred.");
		case WBEM_E_ENCRYPTED_CONNECTION_REQUIRED: throw Exception("User attempted to set a computer name or domain without an encrypted connection.");
		case WBEM_E_PROVIDER_TIMED_OUT: throw Exception("A provider failed to report results within the specified timeout.");
		case WBEM_E_NO_KEY: throw Exception("User attempted to put an instance with no defined key.");
		case WBEM_E_PROVIDER_DISABLED: throw Exception("User attempted to register a provider instance but the COM server for the provider instance was unloaded.");
		case WBEMESS_E_REGISTRATION_TOO_BROAD: throw Exception("Provider registration overlaps with the system event domain.");
		case WBEMESS_E_REGISTRATION_TOO_PRECISE: throw Exception("A WITHIN clause was not used in this query.");
		case WBEMESS_E_AUTHZ_NOT_PRIVILEGED: throw Exception("This computer does not have the necessary domain permissions to support the security functions that relate to the created subscription instance. Contact the Domain Administrator to get this computer added to the Windows Authorization Access Group.");
		case 0x80043001: throw Exception("Reserved for future use.");		// WBEM_E_RETRY_LATER
		case 0x80043002: throw Exception("Reserved for future use.");		// WBEM_E_RESOURCE_CONTENTION
		case WBEMMOF_E_EXPECTED_QUALIFIER_NAME: throw Exception("Expected a qualifier name.");
		case WBEMMOF_E_EXPECTED_SEMI: throw Exception("Expected semicolon or '='.");
		case WBEMMOF_E_EXPECTED_OPEN_BRACE: throw Exception("Expected an opening brace.");
		case WBEMMOF_E_EXPECTED_CLOSE_BRACE: throw Exception("Missing closing brace or an illegal array element.");
		case WBEMMOF_E_EXPECTED_CLOSE_BRACKET: throw Exception("Expected a closing bracket.");
		case WBEMMOF_E_EXPECTED_CLOSE_PAREN: throw Exception("Expected closing parenthesis.");
		case WBEMMOF_E_ILLEGAL_CONSTANT_VALUE: throw Exception("Numeric value out of range or std::strings without quotes.");
		case WBEMMOF_E_EXPECTED_TYPE_IDENTIFIER: throw Exception("Expected a type identifier.");
		case WBEMMOF_E_EXPECTED_OPEN_PAREN: throw Exception("Expected an open parenthesis.");
		case WBEMMOF_E_UNRECOGNIZED_TOKEN: throw Exception("Unexpected token in the file.");
		case WBEMMOF_E_UNRECOGNIZED_TYPE: throw Exception("Unrecognized or unsupported type identifier.");
		case WBEMMOF_E_EXPECTED_PROPERTY_NAME: throw Exception("Expected property or method name.");
		case WBEMMOF_E_TYPEDEF_NOT_SUPPORTED: throw Exception("Typedefs and enumerated types are not supported.");
		case WBEMMOF_E_UNEXPECTED_ALIAS: throw Exception("Only a reference to a class object can have an alias value.");
		case WBEMMOF_E_UNEXPECTED_ARRAY_INIT: throw Exception("Unexpected array initialization. Arrays must be declared with [].");
		case WBEMMOF_E_INVALID_AMENDMENT_SYNTAX: throw Exception("Namespace path syntax is not valid.");
		case WBEMMOF_E_INVALID_DUPLICATE_AMENDMENT: throw Exception("Duplicate amendment specifiers.");
		case WBEMMOF_E_INVALID_PRAGMA: throw Exception("#pragma must be followed by a valid keyword.");
		case WBEMMOF_E_INVALID_NAMESPACE_SYNTAX: throw Exception("Namespace path syntax is not valid.");
		case WBEMMOF_E_EXPECTED_CLASS_NAME: throw Exception("Unexpected character in class name must be an identifier.");
		case WBEMMOF_E_TYPE_MISMATCH: throw Exception("The value specified cannot be made into the appropriate type.");
		case WBEMMOF_E_EXPECTED_ALIAS_NAME: throw Exception("Dollar sign must be followed by an alias name as an identifier.");
		case WBEMMOF_E_INVALID_CLASS_DECLARATION: throw Exception("Class declaration is not valid.");
		case WBEMMOF_E_INVALID_INSTANCE_DECLARATION: throw Exception("The instance declaration is not valid. It must start with \"instance of\"");
		case WBEMMOF_E_EXPECTED_DOLLAR: throw Exception("Expected dollar sign. An alias in the form \"$name\" must follow the \"as\" keyword.");
		case WBEMMOF_E_CIMTYPE_QUALIFIER: throw Exception("\"CIMTYPE\" qualifier cannot be specified directly in a MOF file. Use standard type notation.");
		case WBEMMOF_E_DUPLICATE_PROPERTY: throw Exception("Duplicate property name was found in the MOF.");
		case WBEMMOF_E_INVALID_NAMESPACE_SPECIFICATION: throw Exception("Namespace syntax is not valid. References to other servers are not allowed.");
		case WBEMMOF_E_OUT_OF_RANGE: throw Exception("Value out of range.");
		case WBEMMOF_E_INVALID_FILE: throw Exception("The file is not a valid text MOF file or binary MOF file.");
		case WBEMMOF_E_ALIASES_IN_EMBEDDED: throw Exception("Embedded objects cannot be aliases.");
		case WBEMMOF_E_NULL_ARRAY_ELEM: throw Exception("NULL elements in an array are not supported.");
		case WBEMMOF_E_DUPLICATE_QUALIFIER: throw Exception("Qualifier was used more than once on the object.");
		case WBEMMOF_E_EXPECTED_FLAVOR_TYPE: throw Exception("Expected a flavor type such as ToInstance, ToSubClass, EnableOverride, or DisableOverride.");
		case WBEMMOF_E_INCOMPATIBLE_FLAVOR_TYPES: throw Exception("Combining EnableOverride and DisableOverride on same qualifier is not legal.");
		case WBEMMOF_E_MULTIPLE_ALIASES: throw Exception("An alias cannot be used twice.");
		case WBEMMOF_E_INCOMPATIBLE_FLAVOR_TYPES2: throw Exception("Combining Restricted, and ToInstance or ToSubClass is not legal.");
		case WBEMMOF_E_NO_ARRAYS_RETURNED: throw Exception("Methods cannot return array values.");
		case WBEMMOF_E_MUST_BE_IN_OR_OUT: throw Exception("Arguments must have an In or Out qualifier.");
		case WBEMMOF_E_INVALID_FLAGS_SYNTAX: throw Exception("Flags syntax is not valid.");
		case WBEMMOF_E_EXPECTED_BRACE_OR_BAD_TYPE: throw Exception("The final brace and semi-colon for a class are missing.");
		case WBEMMOF_E_UNSUPPORTED_CIMV22_QUAL_VALUE: throw Exception("A CIM version 2.2 feature is not supported for a qualifier value.");
		case WBEMMOF_E_UNSUPPORTED_CIMV22_DATA_TYPE: throw Exception("The CIM version 2.2 data type is not supported.");
		case WBEMMOF_E_INVALID_DELETEINSTANCE_SYNTAX: throw Exception("The delete instance syntax is not valid. It should be #pragma DeleteInstance(\"instancepath\", FAIL|NOFAIL)");
		case WBEMMOF_E_INVALID_QUALIFIER_SYNTAX: throw Exception("The qualifier syntax is not valid. It should be qualifiername:type=value,scope(class|instance), flavorname.");
		case WBEMMOF_E_QUALIFIER_USED_OUTSIDE_SCOPE: throw Exception("The qualifier is used outside of its scope.");
		case WBEMMOF_E_ERROR_CREATING_TEMP_FILE: throw Exception("Error creating temporary file. The temporary file is an intermediate stage in the MOF compilation.");
		case WBEMMOF_E_ERROR_INVALID_INCLUDE_FILE: throw Exception("A file included in the MOF by the preprocessor command #include is not valid.");
		case WBEMMOF_E_INVALID_DELETECLASS_SYNTAX: throw Exception("The syntax for the preprocessor commands #pragma deleteinstance or #pragma deleteclass is not valid.");

		default: break;
		}

		#if 0
		HRESULT hr;
		string description;
		string source;
		string helpfile;

		// TODO: Must check for rich error support on the interface before using ::GetErrorInfo(), otherwise we might
		// be retrieving a stale error from a previous interface.
		if (pObject != nullptr)
		{
			com_ptr<ISupportErrorInfo> pSupportErrorInfo;
			HRESULT hr = pObject->QueryInterface(__uuidof(ISupportErrorInfo), pSupportErrorInfo.attachment());
			if (SUCCEEDED(hr))
			{
				hr = pSupportErrorInfo->InterfaceSupportsErrorInfo(__uuidof(T));
				if (hr == S_OK)
				{
					// This interface supports rich errors.  					

					com_ptr<IErrorInfo> pErrInfo;
					hr = ::GetErrorInfo(0, &pErrInfo);
					if (hr != S_OK) throw COMException(LastError);

					BSTR bstrDescription;
					hr = pErrInfo->GetDescription(&bstrDescription);
					if (FAILED(hr)) throw COMException(LastError);
					if (bstrDescription != NULL)
					{
						_bstr_t wrapperDesc(bstrDescription, false);
						description = to_string(wstring((const wchar_t*)wrapperDesc));
					}

					BSTR bstrSource;
					hr = pErrInfo->GetSource(&bstrSource);
					if (FAILED(hr)) throw COMException(LastError);
					if (bstrSource != NULL)
					{
						_bstr_t wrapperSource(bstrSource, false);
						source = to_string(wstring((const wchar_t*)wrapperSource));
					}

					BSTR bstrHelpFile;
					hr = pErrInfo->GetHelpFile(&bstrHelpFile);
					if (FAILED(hr)) throw COMException(LastError);
					if (bstrHelpFile != NULL)
					{
						_bstr_t wrapperHelpFile(bstrHelpFile, false);
						helpfile = to_string(wstring((const wchar_t*)wrapperHelpFile));
					}

					if (description.length() == 0)
					{
						UInt32 Win32Code = 0;
						if ((hr & 0xFFFF0000) == MAKE_HRESULT(SEVERITY_ERROR, FACILITY_WIN32, 0))
						{
							Win32Code = HRESULT_CODE(hr);

							TCHAR* pszBuffer;
							if (::FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_IGNORE_INSERTS,
								nullptr, Win32Code,
								MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPTSTR)&pszBuffer, sizeof(TCHAR*), nullptr) != 0
								&& pszBuffer != nullptr)
							{
								osstring os_msg = pszBuffer;
								::LocalFree(pszBuffer);
								description = to_string(os_msg);
							}
						}
					}
		#endif

		char tmp[256];
		sprintf_s(tmp, sizeof(tmp), "COM Error 0x%08X has occurred", LastError);
		string msg = tmp;
		//if (description.length() > 0) msg = msg + ": " + description; else msg = msg + ".";
		//if (source.length() > 0) msg = msg + "\nSource: " + source;
		//if (helpfile.length() > 0) msg = msg + "\nHelp file: " + helpfile;

		throw COMException(msg, LastError);

		/*
		_com_error err(LastError);
		LPCTSTR errMsg = err.ErrorMessage();
		throw COMException(to_string(errMsg), LastError);
		*/
	}
	#endif	// _WINDOWS
}

#endif	// __WBExceptions_h__

//	End of Exceptions.h


