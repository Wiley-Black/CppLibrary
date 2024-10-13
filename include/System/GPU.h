/////////
//  GPU.h
//  Copyright (C) 2021 by Wiley Black
/////////

#ifndef __wbGPU_h__
#define __wbGPU_h__

#include "../wbFoundation.h"
#include "../Foundation/Exceptions.h"

#ifdef CUDA_Support
#include <cuda.h>
#include <driver_types.h>			// For cudaError_t.
#include <cuda_runtime_api.h>		// For cudaGetErrorString().

#if (CUDART_VERSION >= 12000)			// CUDA 12+
#include <cuComplex.h>
#endif

	// Note: I think the following is unnecessary for CUDA 11.2, but seems to be needed for CUDA 10.2.  It is harmless for CUDA 11.2 so included in all cases.
#pragma comment(lib, "cuda.lib")	// Force linkage to cuda.lib to get access to driver API, i.e. cuGetErrorString() and cuCtxGetDevice().
#pragma comment(lib, "cudart.lib")

#ifdef NPP_Support
/** NPP (NVidia Performance Primitives) Library **/
#include "nppdefs.h"
#include "nppcore.h"
#include "nppi.h"
#include "npps.h"
#include "npp.h"

/** List for CUDA 10.2 **/
#if (CUDART_VERSION == 10020)			// CUDA 10.2
#pragma comment(lib, "nppial.lib")
#pragma comment(lib, "nppicc.lib")
#pragma comment(lib, "nppicom.lib")
#pragma comment(lib, "nppidei.lib")
#pragma comment(lib, "nppif.lib")
#pragma comment(lib, "nppig.lib")
#pragma comment(lib, "nppim.lib")
#pragma comment(lib, "nppist.lib")
#pragma comment(lib, "nppisu.lib")
#pragma comment(lib, "nppitc.lib")
#elif (CUDART_VERSION >= 11000)			// CUDA 11+
#pragma comment(lib, "nppc.lib")
#pragma comment(lib, "nppial.lib")
#pragma comment(lib, "nppicc.lib")
#pragma comment(lib, "nppidei.lib")
#pragma comment(lib, "nppif.lib")
#pragma comment(lib, "nppig.lib")
#pragma comment(lib, "nppim.lib")
#pragma comment(lib, "nppist.lib")
#pragma comment(lib, "nppisu.lib")
#pragma comment(lib, "nppitc.lib")
#pragma comment(lib, "npps.lib")
#endif

//#pragma comment(lib, "nppi.lib")
#endif		// NPP_Support

#ifdef CUB_Support
#ifdef __CUDACC__
#include <cub/cub.cuh>
#endif
#endif

#ifdef cuFFT_Support
#include "cufft.h"
#pragma comment(lib, "cufft.lib")
#endif		// cuFFT_Support

#ifdef NVTX_Enable
#include "Processing/CUDA/NVTX.h"
#endif

#endif		// CUDA_Support

namespace wb
{
	namespace cuda
	{
		using namespace wb;		
		
		#pragma region "numeric_limits<T>"

		/** numeric_limits for device (std::numeric_limits isn't available under CUDA compilation, so we provide it here) **/

		#undef min
		#undef max		

		#ifdef __CUDACC__
		#define __nlc_keywords__ __host__ __device__ __forceinline__
		#else
		#define __nlc_keywords__ inline
		#endif		

		// Note: can't be used for floating-point types (because the C++ standard does not allow template non-type arguments
		// of floating-point types), and for integral types lowest = min in all cases.
		template<typename T, T min_value, T max_value> class numeric_limits_core
		{
		public:			
			static constexpr __nlc_keywords__ T min() noexcept { return min_value; }
			static constexpr __nlc_keywords__ T lowest() noexcept { return min_value; }
			static constexpr __nlc_keywords__ T max() noexcept { return max_value; }
		};

		template<typename T> class numeric_limits { };
		template<> class numeric_limits<bool> : public numeric_limits_core<bool, false, true> { };
		template<> class numeric_limits<Int8> : public numeric_limits_core<Int8, (-127 - 1), 127> { };
		template<> class numeric_limits<Int16> : public numeric_limits_core<Int16, (-32767 - 1), 32767> { };
		template<> class numeric_limits<Int32> : public numeric_limits_core<Int32, (-2147483647 - 1), 2147483647> { };
		template<> class numeric_limits<Int64> : public numeric_limits_core<Int64, (-9223372036854775807LL - 1), 9223372036854775807LL> { };
		template<> class numeric_limits<UInt8> : public numeric_limits_core <UInt8, 0, 255> { };
		template<> class numeric_limits<UInt16> : public numeric_limits_core <UInt16, 0, 65535> { };
		template<> class numeric_limits<UInt32> : public numeric_limits_core <UInt32, 0, 4294967295U> { };
		template<> class numeric_limits<UInt64> : public numeric_limits_core <UInt64, 0, 18446744073709551615ULL> { };
		template<> class numeric_limits<float> {
		public:
			static constexpr __nlc_keywords__ float min() noexcept { return FLT_MIN; }
			static constexpr __nlc_keywords__ float lowest() noexcept { return -FLT_MAX; }
			static constexpr __nlc_keywords__ float max() noexcept { return FLT_MAX; }
		};
		template<> class numeric_limits<double> {
		public:
			static constexpr __nlc_keywords__ double min() noexcept { return DBL_MIN; }
			static constexpr __nlc_keywords__ double lowest() noexcept { return -DBL_MAX; }
			static constexpr __nlc_keywords__ double max() noexcept { return DBL_MAX; }
		};
		template<> class numeric_limits<long double> {
		public:
			static constexpr __nlc_keywords__ long double min() noexcept { return LDBL_MIN; }
			static constexpr __nlc_keywords__ long double lowest() noexcept { return -LDBL_MAX; }
			static constexpr __nlc_keywords__ long double max() noexcept { return LDBL_MAX; }
		};

		#pragma endregion

		#pragma region "Exceptions"

		/** Exceptions **/
		
		class GPUException : public ::wb::Exception {
		public:
			GPUException() : Exception(S("A GPU-related error has occurred.")) { }
			GPUException(const char* const& message) : Exception(message) { }
			GPUException(const string& message) : Exception(message) { }
			GPUException(const GPUException& right) : Exception(right) { }
			GPUException(GPUException&& from) noexcept : Exception(from) { }
			GPUException& operator=(const GPUException& right) { Exception::operator=(right); return *this; }
		};

		#pragma endregion

		#ifdef CUDA_Support

		#pragma region "Definitions"

		struct ComputeCapability
		{
			int Major, Minor;

			ComputeCapability() { Major = Minor = 0; }
			ComputeCapability(int Major, int Minor) {
				this->Major = Major; this->Minor = Minor;
			}
		};

		/// <summary>
		/// divup() performs a simple division, but always rounding up instead of down.  For example,
		/// a regular integer divisions of 5/5, 6/5, or 9/5 would all return 1.  divup(5,5) returns 1
		/// but divup(6,5) and divup(9,5) both return 2.  divup() is useful when you need to ensure
		/// that you match exactly or overallocate but never underallocate.
		/// </summary>
		inline int divup(const int numerator, const int denominator) { return (numerator + (denominator - 1)) / denominator; }

		#pragma endregion

		#pragma region "Error handling"

		/// <summary>
		/// Confirm that a driver API has succeeded or throw an exception with an
		/// error message.
		/// </summary>
		/// <param name="cuda_error_code"></param>
		inline void Throwable(CUresult cu_error_code, const char* pszSourceFile, int nSourceLine)
		{
			if (cu_error_code == CUDA_SUCCESS) return;
			const char* pStr;
			try
			{				
				CUresult retrieval_code = cuGetErrorString(cu_error_code, &pStr);
				if (retrieval_code != CUDA_SUCCESS) throw GPUException();				
			}
			catch (std::exception& ex)
			{
				throw GPUException("A GPU-related error has occurred and error retrieval was unavailable: " + string(ex.what()) + "  Original Source: " + string(pszSourceFile) + ":" + std::to_string(nSourceLine) + "  Original Code: " + std::to_string((int)cu_error_code));
			}
			throw GPUException("GPU Error: " + string(pStr) + "  Source: " + string(pszSourceFile) + ":" + std::to_string(nSourceLine) + "  Code: " + std::to_string((int)cu_error_code));
		}

		/// <summary>
		/// Confirm that a CUDA Runtime API has succeeded or throw an exception with
		/// an error message.
		/// </summary>
		/// <param name="cuda_error_code"></param>
		inline void Throwable(cudaError_t cuda_error_code, const char* pszSourceFile, int nSourceLine)
		{
			if (cuda_error_code == cudaSuccess) return;
			const char* pStr = cudaGetErrorString(cuda_error_code);				
			// If there is an error, cudaGetErrorString() will return "unrecognized error code".

			throw GPUException("GPU Error: " + string(pStr) + "  Source: " + string(pszSourceFile) + ":" + std::to_string(nSourceLine) + "  Code: " + std::to_string((int)cuda_error_code));
		}

		#ifdef NPP_Support
		inline void Throwable(NppStatus Status, const char* pszSourceFile, int nSourceLine)
		{
			if (Status == 0) return;				// Success.
			const char* pszMessage;
			switch (Status)
			{
			case NPP_NOT_SUPPORTED_MODE_ERROR: pszMessage = "Unsupported mode."; break;
			case NPP_INVALID_HOST_POINTER_ERROR: pszMessage = "Invalid host pointer."; break;
			case NPP_INVALID_DEVICE_POINTER_ERROR: pszMessage = "Invalid device pointer."; break;
			case NPP_LUT_PALETTE_BITSIZE_ERROR: pszMessage = "LUT Palette bitsize invalid."; break;
			case NPP_ZC_MODE_NOT_SUPPORTED_ERROR: pszMessage = "ZeroCrossing mode not supported."; break;
			case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY: pszMessage = "Insufficient compute capability in GPU device."; break;
			case NPP_TEXTURE_BIND_ERROR: pszMessage = "Texture binding error."; break;
			case NPP_WRONG_INTERSECTION_ROI_ERROR: pszMessage = "Wrong intersection ROI."; break;
			case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR: pszMessage = "HAAR Classifier pixel match error."; break;
			case NPP_MEMFREE_ERROR: pszMessage = "Memory release error."; break;
			case NPP_MEMSET_ERROR: pszMessage = "Memory set error."; break;
			case NPP_MEMCPY_ERROR: pszMessage = "Memory copy error."; break;
			case NPP_ALIGNMENT_ERROR: pszMessage = "Alignment error."; break;
			case NPP_CUDA_KERNEL_EXECUTION_ERROR: pszMessage = "CUDA Kernel execution failure."; break;
			case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR: pszMessage = "Unsupported round mode."; break;
			case NPP_QUALITY_INDEX_ERROR: pszMessage = "Image pixels are constant for quality index."; break;
			case NPP_RESIZE_NO_OPERATION_ERROR: pszMessage = "One of the output image dimensions is less than 1 pixel."; break;
			case NPP_OVERFLOW_ERROR: pszMessage = "Number overflows the upper or lower limit of the data type."; break;
			case NPP_NOT_EVEN_STEP_ERROR: pszMessage = "Step value is not pixel multiple."; break;
			case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR: pszMessage = "Number of levels for histogram is less than 2"; break;
			case NPP_LUT_NUMBER_OF_LEVELS_ERROR: pszMessage = "Number of levels for LUT is less than 2."; break;
			case NPP_CORRUPTED_DATA_ERROR: pszMessage = "Processed data is corrupted."; break;
			case NPP_CHANNEL_ORDER_ERROR: pszMessage = "Wrong order of the destination channels."; break;
			case NPP_ZERO_MASK_VALUE_ERROR: pszMessage = "All values of the mask are zero."; break;
			case NPP_QUADRANGLE_ERROR: pszMessage = "The quadrangle is nonconvex or degenerates into triangle, line or point."; break;
			case NPP_RECTANGLE_ERROR: pszMessage = "Size of the rectangle region is less than or equal to 1."; break;
			case NPP_COEFFICIENT_ERROR: pszMessage = "Unallowable values of the transformation coefficients."; break;
			case NPP_NUMBER_OF_CHANNELS_ERROR: pszMessage = "Bad or unsupported number of channels."; break;
			case NPP_COI_ERROR: pszMessage = "Channel of interest is not 1, 2, or 3."; break;
			case NPP_DIVISOR_ERROR: pszMessage = "Divisor is equal to zero."; break;
			case NPP_CHANNEL_ERROR: pszMessage = "Illegal channel index."; break;
			case NPP_STRIDE_ERROR: pszMessage = "Stride is less than the row length."; break;
			case NPP_ANCHOR_ERROR: pszMessage = "Anchor point is outside mask."; break;
			case NPP_MASK_SIZE_ERROR: pszMessage = "Lower bound is larger than upper bound."; break;
			case NPP_RESIZE_FACTOR_ERROR: pszMessage = "Resize factor invalid."; break;
			case NPP_INTERPOLATION_ERROR: pszMessage = "Interpolation failure."; break;
			case NPP_MIRROR_FLIP_ERROR: pszMessage = "Mirror flip failed."; break;
			case NPP_MOMENT_00_ZERO_ERROR: pszMessage = "Moment 00 zero error."; break;
			case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR: pszMessage = "Threshold negative level."; break;
			case NPP_THRESHOLD_ERROR: pszMessage = "Threshold error."; break;
			case NPP_CONTEXT_MATCH_ERROR: pszMessage = "Context match error."; break;
			case NPP_FFT_FLAG_ERROR: pszMessage = "FFT Flag error."; break;
			case NPP_FFT_ORDER_ERROR: pszMessage = "FFT Order error."; break;
			case NPP_STEP_ERROR: pszMessage = "Step is less than or equal to zero."; break;
			case NPP_SCALE_RANGE_ERROR: pszMessage = "Scale range error."; break;
			case NPP_DATA_TYPE_ERROR: pszMessage = "Data type error."; break;
			case NPP_OUT_OFF_RANGE_ERROR: pszMessage = "Out of range error."; break;
			case NPP_DIVIDE_BY_ZERO_ERROR: pszMessage = "Divide by zero error."; break;
			case NPP_MEMORY_ALLOCATION_ERR: pszMessage = "Memory allocation error."; break;
			case NPP_NULL_POINTER_ERROR: pszMessage = "Null pointer error."; break;
			case NPP_RANGE_ERROR: pszMessage = "Out of range."; break;
			case NPP_SIZE_ERROR: pszMessage = "Size."; break;
			case NPP_BAD_ARGUMENT_ERROR: throw ArgumentException();
			case NPP_NO_MEMORY_ERROR: throw OutOfMemoryException();
			case NPP_NOT_IMPLEMENTED_ERROR: throw NotImplementedException();
			default:
			case NPP_ERROR: pszMessage = "Error."; break;
			case NPP_ERROR_RESERVED: pszMessage = "Reserved."; break;
			case NPP_NO_OPERATION_WARNING: pszMessage = "Indicates that no operation was performed."; break;
			case NPP_DIVIDE_BY_ZERO_WARNING: pszMessage = "Divisor is zero however does not terminate the execution."; break;
			case NPP_AFFINE_QUAD_INCORRECT_WARNING: pszMessage = "Indicates that the quadrangle passed to one of affine warping functions doesn’PixelType have necessary properties.  First 3 vertices are used, the fourth vertex discarded."; break;
			case NPP_WRONG_INTERSECTION_ROI_WARNING: pszMessage = "The given ROI has no interestion with either the source or destination ROI.  Thus no operation was performed."; break;
			case NPP_WRONG_INTERSECTION_QUAD_WARNING: pszMessage = "The given quadrangle has no intersection with either the source or destination ROI.  Thus no operation was performed."; break;
			case NPP_DOUBLE_SIZE_WARNING: pszMessage = "Image size isn’PixelType multiple of two.  Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing."; break;
			case NPP_MISALIGNED_DST_ROI_WARNING: pszMessage = "Speed reduction due to uncoalesced memory accesses warning."; break;
			}
			throw Exception("NPP (CUDA) Error: " + string(pszMessage) + "  Source: " + string(pszSourceFile) + ":" + std::to_string(nSourceLine) + "  Code: " + std::to_string((int)Status));
		}
		#endif

		#ifdef cuFFT_Support
		inline void Throwable(cufftResult_t cufft_error_code, const char* pszSourceFile, int nSourceLine)
		{
			if (cufft_error_code == CUFFT_SUCCESS) return;
			const char* pszMsg = nullptr;
			switch (cufft_error_code)
			{				
			case CUFFT_INVALID_PLAN: pszMsg = "invalid plan"; break;
			case CUFFT_ALLOC_FAILED: pszMsg = "allocation failed"; break;
			case CUFFT_INVALID_TYPE: pszMsg = "invalid type"; break;
			case CUFFT_INVALID_VALUE: pszMsg = "invalid value"; break;
			case CUFFT_INTERNAL_ERROR: pszMsg = "internal error"; break;
			case CUFFT_EXEC_FAILED: pszMsg = "execution failed"; break;
			case CUFFT_SETUP_FAILED: pszMsg = "setup failed"; break;
			case CUFFT_INVALID_SIZE: pszMsg = "invalid size"; break;
			case CUFFT_UNALIGNED_DATA: pszMsg = "unaligned data"; break;
			case CUFFT_INCOMPLETE_PARAMETER_LIST: pszMsg = "incomplete parameter list"; break;
			case CUFFT_INVALID_DEVICE: pszMsg = "invalid device"; break;
			case CUFFT_PARSE_ERROR: pszMsg = "parsing error"; break;
			case CUFFT_NO_WORKSPACE: pszMsg = "no workspace"; break;
			case CUFFT_NOT_IMPLEMENTED: pszMsg = "not implemented"; break;
			case CUFFT_LICENSE_ERROR: pszMsg = "license error"; break;
			case CUFFT_NOT_SUPPORTED: pszMsg = "not supported"; break;
			default: pszMsg = "unrecognized error"; break;
			}
			throw GPUException("FFT (CUDA) Error: " + string(pszMsg) + "  Source: " + string(pszSourceFile) + ":" + std::to_string(nSourceLine) + "  Code: " + std::to_string((int)cufft_error_code));
		}
		#endif

		#define cudaThrowable(err_code)		::wb::cuda::Throwable(err_code, __FILE__, __LINE__)			

		#pragma endregion

		#pragma region "Scratch Memory Buffers"

		namespace memory
		{
			class DeviceScratchBuffer
			{
				void DoFree()
				{
					assert(m_pData != nullptr);
					cudaThrowable(cudaFree(m_pData));
					m_pData = nullptr;
				}

			public:

				byte*		m_pData;
				size_t		m_Size;

				byte* GetDevicePtr() { return m_pData; }
				size_t GetSize() { return m_Size; }

				bool IsAllocated(size_t nSize) const
				{
					return (m_pData != nullptr && m_Size >= nSize);
				}

				void Allocate(size_t nSize)
				{
					/** Re-use as-is? **/
					if (IsAllocated(nSize)) return;     // Can use as-is.					

					/** Reallocate, resize, or release? **/
					if (m_pData != nullptr)
					{
						// We don't have a reallocate option in the cuda library for device
						// memory.  Since we need to grow the data buffer, our only option
						// remaining is to free up the memory and allocate something new.
						DoFree();
					}

					/** Allocate new memory **/
					m_Size = nSize;
					cudaThrowable(cudaMalloc(&m_pData, nSize));
					if (m_pData == nullptr) throw OutOfMemoryException();
				}

				DeviceScratchBuffer()
				{
					m_pData = nullptr;
					m_Size = 0;
				}

				DeviceScratchBuffer(DeviceScratchBuffer&) = delete;
				DeviceScratchBuffer& operator=(DeviceScratchBuffer&) = delete;

				DeviceScratchBuffer(DeviceScratchBuffer&& mv)
				{
					m_pData = mv.m_pData;
					mv.m_pData = nullptr;
					m_Size = mv.m_Size;
				}

				DeviceScratchBuffer& operator=(DeviceScratchBuffer&& mv)
				{
					if (m_pData != nullptr) {
						DoFree(); m_pData = nullptr;
					}
					m_pData = mv.m_pData;
					mv.m_pData = nullptr;
					m_Size = mv.m_Size;
					return *this;
				}

				~DeviceScratchBuffer()
				{
					if (m_pData != nullptr) DoFree();
					m_Size = 0;
				}
			};
		}

		#pragma endregion

		#pragma region "Stream and Device Management"

		/// <summary>
		/// GPUSystemInfo provides reference-based access to GPU device properties without needing to call 
		/// cudaGetDeviceProperties() routinely or having to copy the somewhat large device_properties structure.
		/// </summary>
		class GPUSystemInfo
		{
		private:
			std::vector<cudaDeviceProp>		device_properties;			// Vector index matches device ordinal.

			/// <summary>
			/// cudaMallocPitch() chooses a pitch for the device based on the width of the image and the GPU characteristics, or so I assume.
			/// cudaMallocPitch() provides a mapping of the power-of-2 width in bytes to the pitch returned by cudaMallocPitch().  If the pitch exactly
			/// matched the width in bytes, then the mapping would be:
			///		pitch_map[1]:	2
			///		pitch_map[2]:	4
			///		pitch_map[3]:	8
			/// 
			/// However, for example, if the minimum pitch on a device is 512 bytes, then the mapping would look like:
			///		pitch_map[7]:	512
			///		pitch_map[8]:	512
			///		pitch_map[9]:	512
			///		pitch_map[10]:	1024
			/// 
			/// The pitch map starts at 2^0 and maxes out at 2^12 (4096), and the GetOptimalPitch() function will account for all these factors.
			/// The outer vector contains one vector per GPU present.
			/// </summary>
			std::vector<std::vector<size_t>>	m_PitchMap;

		public:
			GPUSystemInfo()
			{				
				int device_count = 0;
				cudaThrowable(cudaGetDeviceCount(&device_count));
				for (int ii = 0; ii < device_count; ii++)
				{
					cudaDeviceProp dev_prop;
					cudaThrowable(cudaGetDeviceProperties(&dev_prop, ii));
					device_properties.push_back(dev_prop);

					cudaThrowable(cudaSetDevice(ii));

					// Perform test memory allocations with cudaMemoryPitch() to understand its results:
					std::vector<size_t> device_pitch_map;
					for (int iPow2 = 0; iPow2 < 12; iPow2++)
					{
						int width_in_bytes = 1 << iPow2;
						void* pDevPtr = nullptr;
						size_t pitch = 0;
						cudaThrowable(cudaMallocPitch(&pDevPtr, &pitch, width_in_bytes, 4));
						cudaThrowable(cudaFree(pDevPtr));
						device_pitch_map.push_back(pitch);						
					}
					m_PitchMap.push_back(device_pitch_map);
				}

				cudaThrowable(cudaSetDevice(0));
			}

			size_t	GetOptimalPitch(int iDevice, unsigned int width_in_bytes)
			{
				if (width_in_bytes == 0) return 0;
				int msb = 0;
				unsigned int wib = width_in_bytes;
				while (wib != 0) {
					wib >>= 1;
					msb++;
				}
				msb--;
				// msb is now the index of the most significant bit, with 0 being the lowest bit, 1 being 10b, 2 being 100b, etc.
				if (width_in_bytes != (1 << msb)) msb++;
				// now in cases where width_in_bytes was not a power-of-2, msb has been increased by one to cover it.
				auto& device_pitch_map = m_PitchMap[iDevice];
				if (msb >= device_pitch_map.size()) return (((size_t)1) << msb);
				return device_pitch_map[msb];
			}

			const cudaDeviceProp& GetDeviceProperties(int cuda_device_id) const
			{
				if (cuda_device_id < 0 || cuda_device_id >= device_properties.size())
					throw ArgumentException("CUDA device id outside enumerated device listing range.");
				return device_properties[cuda_device_id];
			}

			static void GetDeviceMemoryUsage(UInt64& FreeMemory, UInt64& TotalMemory)
			{
				size_t free_byte;
				size_t total_byte;
				cudaThrowable(cudaMemGetInfo(&free_byte, &total_byte));
				FreeMemory = free_byte;
				TotalMemory = total_byte;
			}
		};

		/// <summary>
		/// The GPUStream class abstracts a shared pointer to a cudaStream_t as well as keeping track
		/// of the properties of the stream/device.  These are properties that can be helpful in 
		/// optimizing the kernels being launched.  For example, GPUStream provides convenient access 
		/// to  device_properties.maxThreadsPerBlock, which can be used in setting up the thread pattern
		/// for a kernel.
		/// 
		/// To use GPUStream, first create a new stream:
		///		
		///		auto my_stream = GPUStream::New();
		/// 
		/// This stream can then be sent to functions that need the GPUStream or a cudaStream_t.  The
		/// GPUStream object is a wrapper around std::shared_ptr&lt;GPUStreamData&gt;, and the stream
		/// is not destroyed until all references have expired.  Using a shared_ptr means that the passing
		/// of the stream information among different functions is quick, and the full information about
		/// the stream and device is available through the pointer.
		/// 
		/// The static function GPUStream::None() can be used as a default parameter, but it simply
		/// prevents access to device or stream properties and will throw an exception on attempts to
		/// access the properties.
		/// 
		/// The New() call can also accept a string that identifies the stream in NVIDIA Nsight Tools when
		/// NVTX_Enable is defined.  It takes no resources when NVTX_Enable is disabled.
		/// </summary>
		class GPUStream
		{
			struct GPUStreamData
			{			
				cudaStream_t						stream;
				int									device;
				bool								responsible;
				cudaDeviceProp						properties;

				/// <summary>
				/// Returns a vector of scratch memory buffers associated with this thread.  As an optimization, when
				/// more than one scratch buffer is needed, use the lowest index scratch buffer in the vector for the
				/// largest buffer.  By doing this consistently, all users of the scratch buffers will follow the same
				/// convention and will avoid unnecessary memory allocation.
				/// </summary>
				vector<memory::DeviceScratchBuffer>	scratch;

				GPUStreamData(cudaStream_t fromStream = nullptr, bool responsible_ = true) : stream(fromStream), responsible(responsible_)
				{
					if (stream == nullptr)
					{
						cudaThrowable(cudaGetDevice(&device));
					}
					else
					{
						CUcontext context;
						cudaThrowable(cuStreamGetCtx(stream, &context));

						// Note: cuCtxPushCurrent() may return error codes from previous, asynchronous launches.
						cudaThrowable(cuCtxPushCurrent(context));
						cudaThrowable(cuCtxGetDevice(&device));			// Retrieves the device's ordinal.
						cudaThrowable(cuCtxPopCurrent(&context));
					}

					cudaThrowable(cudaGetDeviceProperties(&properties, device));
				}

				GPUStreamData(GPUStreamData&) = delete;
				GPUStreamData(GPUStreamData&&) = delete;
				GPUStreamData& operator=(GPUStreamData&) = delete;
				GPUStreamData& operator=(GPUStreamData&&) = delete;

				~GPUStreamData()
				{
					if (stream != nullptr)
					{
						if (responsible)
							cudaThrowable(cudaStreamDestroy(stream));
						stream = nullptr;
					}
				}
			};

			std::shared_ptr<GPUSystemInfo>	m_pGSI;

		public:		// Should probably be private, but this makes the ostream operator easy.
			std::shared_ptr<GPUStreamData>	m_pData;								

		private:
			GPUStream() { }

		public:			
			static GPUStream New(std::shared_ptr<GPUSystemInfo>& pGSI, const string& sDiagnosticName = "")
			{				
				GPUStream ret;
				ret.m_pGSI = pGSI;
				cudaStream_t stream;
				cudaThrowable(cudaStreamCreate(&stream));
				#ifdef NVTX_Enable
				if (!sDiagnosticName.empty())
					nvtx::SetName(stream, sDiagnosticName.c_str());
				#endif
				ret.m_pData = make_shared<GPUStreamData>(stream);
				return ret;
			}

			static GPUStream Existing(std::shared_ptr<GPUSystemInfo>& pGSI, cudaStream_t from_stream, const string& sDiagnosticName = "")
			{
				GPUStream ret;
				ret.m_pGSI = pGSI;								
#ifdef NVTX_Enable
				if (!sDiagnosticName.empty())
					nvtx::SetName(from_stream, sDiagnosticName.c_str());
#endif
				ret.m_pData = make_shared<GPUStreamData>(from_stream, false);
				return ret;
			}

			static GPUStream None()
			{
				GPUStream ret;
				return ret;
			}

			GPUStream(const GPUStream& cp) = default;
			GPUStream(GPUStream&& mvStream) noexcept = default;
			GPUStream& operator=(const GPUStream&) = default;
			GPUStream& operator=(GPUStream&&) = default;

			const cudaDeviceProp&	GetDeviceProperties() const
			{
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				return m_pData->properties;
			}

			/// <summary>
			/// Returns a vector of scratch memory buffers associated with this thread.  As an optimization, when
			/// more than one scratch buffer is needed, use the lowest index scratch buffer in the vector for the
			/// largest buffer.  By doing this consistently, all users of the scratch buffers will follow the same
			/// convention and will avoid unnecessary memory allocation.
			/// </summary>
			vector<memory::DeviceScratchBuffer>&	GetScratch()
			{
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				return m_pData->scratch;
			}

			int	 GetDeviceId()
			{
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				return m_pData->device;
			}

			bool IsNone() const { return m_pData == nullptr; }

			operator cudaStream_t() const { 
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				return m_pData->stream; 
			}

			GPUSystemInfo& GetGSI()
			{
				if (m_pGSI == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				return *m_pGSI;
			}

			void Synchronize()
			{
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");
				cudaThrowable(cudaStreamSynchronize(m_pData->stream));
			}

			#ifdef NPP_Support
			operator NppStreamContext() const 
			{
				if (m_pData == nullptr) throw NotSupportedException("Cannot access GPUStream created with None().");

				NppStreamContext ret;
				ret.hStream = m_pData->stream;
				ret.nCudaDeviceId = m_pData->device;
				const cudaDeviceProp& dp = m_pData->properties;
				ret.nMultiProcessorCount = dp.multiProcessorCount;
				ret.nMaxThreadsPerMultiProcessor = dp.maxThreadsPerMultiProcessor;
				ret.nMaxThreadsPerBlock = dp.maxThreadsPerBlock;
				ret.nSharedMemPerBlock = dp.sharedMemPerBlock;
				ret.nCudaDevAttrComputeCapabilityMajor = dp.major;
				ret.nCudaDevAttrComputeCapabilityMinor = dp.minor;
				cudaThrowable(cudaStreamGetFlags(m_pData->stream, &ret.nStreamFlags));
				return ret;
			}
			#endif
		};

		inline std::ostream& operator<<(std::ostream& os, const GPUStream& gStream)
		{
			os << "GPUStream[" << gStream.m_pData.use_count() << " references, stream " << (cudaStream_t)gStream << "]";
			return os;
		}

		#pragma endregion

		#else			// CUDA_Support

		// Define some types and constants from CUDA for convenience...
		// 
		//typedef void* cudaStream_t;		

		struct float2 { float x; float y; /*float __cuda_gnu_arm_ice_workaround[0];*/ };
		struct double2 { double x, y; };
		typedef float2 cuFloatComplex;
		typedef double2 cuDoubleComplex;

		typedef enum
		{
			NPPI_INTER_UNDEFINED = 0,
			NPPI_INTER_NN = 1,        /**<  Nearest neighbor filtering. */
			NPPI_INTER_LINEAR = 2,        /**<  Linear interpolation. */
			NPPI_INTER_CUBIC = 4,        /**<  Cubic interpolation. */
			NPPI_INTER_CUBIC2P_BSPLINE,              /**<  Two-parameter cubic filter (B=1, C=0) */
			NPPI_INTER_CUBIC2P_CATMULLROM,           /**<  Two-parameter cubic filter (B=0, C=1/2) */
			NPPI_INTER_CUBIC2P_B05C03,               /**<  Two-parameter cubic filter (B=1/2, C=3/10) */
			NPPI_INTER_SUPER = 8,        /**<  Super sampling. */
			NPPI_INTER_LANCZOS = 16,       /**<  Lanczos filtering. */
			NPPI_INTER_LANCZOS3_ADVANCED = 17,       /**<  Generic Lanczos filtering with order 3. */
			NPPI_SMOOTH_EDGE = (int)0x8000000 /**<  Smooth edge filtering. */
		} NppiInterpolationMode;

		/// <summary>
		/// Provide a placeholder version when not compiling with NVCC.
		/// </summary>
		class GPUSystemInfo
		{
		public:
			GPUSystemInfo() { }			
		};

		/// <summary>
		/// When CUDA_Support is disabled, GPUStream is still defined as an empty placeholder
		/// to make code more consistent where desirable.
		/// </summary>
		class GPUStream
		{
			GPUStream()
			{
			}

		public:			

			static GPUStream New() {
				return GPUStream();
			}

			static GPUStream None() {
				return GPUStream();
			}

			bool IsNone() const { return true; }

			GPUStream(const GPUStream& fromStream) = default;			
			GPUStream(GPUStream&& mvStream) noexcept = default;
			GPUStream& operator=(const GPUStream&) = default;
			GPUStream& operator=(GPUStream&&) = default;
		};

		#endif			// CUDA_Support		

		// The Launch_Kernel() macro has the following format:
		//		First argument:			Kernel function name, i.e. ExampleKernel.
		//		Next 4 arguments:		Provided inside the <<<...>>> syntax of CUDA in sequence.
		//		Additional arguments:	Arguments to the function.
		//
		// The Launch_Kernel() macro accomplishes the following:
		//	1. Launch a kernel
		//	2. If compiling without the NVidia compiler, an error message about the Unsupported_Call_Requires_NVCC 
		//		function is generated.  Source files (including headers) that incorporate a kernel and have 
		//		CUDA_Support enabled can only be included and used from .cu files.		
		//	3. Hides the __CUDACC__ check within the macro so as to make the calling code cleaner and consistent
		//		while providing working Intellisense.
		#ifdef __CUDACC__
		#define Launch_CUDAKernel(Kernel_Name, GridSize, BlockSize, SMEMSize, Stream, ...) { Kernel_Name <<<(GridSize), (BlockSize), (SMEMSize), (Stream)>>> (__VA_ARGS__); }
		#else
		#define Launch_CUDAKernel(Kernel_Name, GridSize, BlockSize, SMEMSize, Stream, ...) { Unsupported_Call_Requires_NVCC(); }
		#endif

		// If you need to call Launch_CUDAKernel on a templated function, then the preprocessor will give you grief using commas.  For example:
		//		Launch_CUDAKernel(Alpha<int, 2>, ...) 
		// will be parsed by the preprocess as two parameters to the macro:		Alpha<int	and		2>.
		// Use of another macro can work around this:
		//		Launch_CUDAKernel(Alpha<int COMMA 2>, ...).
		#define COMMA ,
	}
}

#endif  // __wbGPU_h__

//  End of GPU.h

