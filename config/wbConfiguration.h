/////////
//	Configuration.h
//	Copyright (C) 2014 by Wiley Black
//
//	Adjust settings here to control the configuration of the Wiley Black C++ Library.
////

#ifndef __WBConfiguration_h__
#define __WBConfiguration_h__

/** Use of Standard Template Library (STL) **/
//#define EmulateSTL					// For the rare case where we don't have STL available, my library can provide some parts of it.

/** DeflateStream Implementation Method **/
//#define DeflateStream_ZLibImpl			// Use the efficient zlib implementation, with dependency on the library.
#define DeflateStream_InternImpl		// Use an internal implementation with no further dependencies.

/** Image Implementation Method **/
#define CUDA_Support					// Define to provide CUDA support for image processing.
#define NPP_Support						// Leverage the NVIDIA Performance Primitives (NPP) library.
//#define CUB_Support						// Leverage the CUB template library.
#define cuFFT_Support					// Leverage the cuFFT library.

/** Image Processing Support **/
//#define FITS_Support					// Define to include support for FITS files.  * Not fully implemented yet.

/** Image Processing - File Save/Load Library Choices - define only one **/
//#define LibTIFF_Support
#define FreeImage_Support				// Add FreeImage's x32 or x64 folder to project include and lib paths.

/** CUDA cuDNN (Deep Learning) Support **/
#define cuDNN_Support					// Define to include support for cuDNN.

#endif	// __WBConfiguration_h__

//	End of Configuration.h

