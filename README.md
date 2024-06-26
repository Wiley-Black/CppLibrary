Wiley Black's CppLibrary is my general-purpose C++ library that uses headers only to remain a lightweight dependency.  It features:
 - XML, JSON, and YAML parsers.
 - Utility classes for strings, dates, encoding, and threading that resemble the .NET API while staying unmanaged.
 - Complete date & time classes stored as 64-bit integers and with parsing.
 - Memory management, including ring buffers.
 - Classes to make Sockets-based networking easier.
 - To a large extent, supports both Windows and Linux seemlessly, although the Linux code needs a refresh.
 - CUDA image processing classes- although I might suggest one of the larger libraries out there for this purpose.

Source: https://github.com/Wiley-Black/CppLibrary

# Usage

This is a "header-only" library that can be accessed with just an #include directive and a little configuration.  

Most of my library is designed to match the .NET Framework API as closely as possible in C++, however it is entirely unmanaged code.  Refer to MSDN documentation for a lot of it.  Some classes have additional comments, usually following the XML \<summary\> format that Visual Studio supports.

# Step 1: configuration

Either add the /config sub-folder to your project's include path or make a copy of the wbConfiguration.h in your project.  
	
Modify the wbConfiguration.h file if you want to change the project configuration.
	
# Step 2: include

Add the /include sub-folder to your project's include path.

Choose one of the following three header files to #include from your code:
	wbFoundation.h		..provides the smallest footprint version of the library.
	wbCore.h			..provides most of the broad functionality of the library, 
						  and the generally common stuff.
	wbComplete.h		..includes everything.

Include the chosen header file before other headers, especially before Windows header files.  This is important for Core.h because 
the Sockets functionality will influence the dependencies that Windows header files will bring in.

If using Foundation or Core, you can still include additional specific headers from the components to add more specific functionality.

# Step 3: PrimaryModule, if needed

If you receive any linker errors, then go to just one of the .cpp files in your project and add the following definition at the top:

#define PrimaryModule

Then provide the #include from step 2.  The "PrimaryModule" definition enables the header files to capture static variables into the .cpp file, essentially 
providing the "non-header" portion of the library.  It must be defined in only one of your .cpp files so as not to duplicate any static variables or 
definitions across compilation units.  The library minimizes its use.

# Step 4: CUDA, if needed

If you are utilizing CUDA_Support and the image processing libraries, make sure that the CUDA Toolkit is installed on your
system.  Go to your project in Visual Studio, right-click and under Build Dependencies, select Build Customizations.  Check
the box for the CUDA toolkit.

Currently built against CUDA 11.4.2.
