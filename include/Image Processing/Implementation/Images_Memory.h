/////////
//	Images_Memory.h
/////////

#ifndef __WBImages_Memory_h__
#define __WBImages_Memory_h__

#include "../../wbFoundation.h"
#include "../../wbCore.h"

#include "../../System/GPU.h"			// Includes cuda.h, if supported.  Defines empty but convenient GPUStream if not supported.
#include "PixelType_Primitives.h"
#include "Images_Kernels.h"

#ifdef FreeImage_Support
#pragma comment(lib, "FreeImage.lib")
#include "FreeImage.h"
#endif

namespace wb
{
	namespace images
	{
		/** References **/
		using namespace wb::cuda;		

		#pragma region "Helpers"

		template<typename PixelType> void CopyMemory2D(PixelType* pDst, int DstStride, PixelType* pSrc, int SrcStride, int Width, int Height)
		{
			if (SrcStride == DstStride && SrcStride == Width * sizeof(PixelType))
			{
				memcpy_s(pDst, DstStride * Height, pSrc, SrcStride * Height);
			}
			else
			{
				assert(DstStride >= Width * sizeof(PixelType));
				assert(SrcStride >= Width * sizeof(PixelType));

				// TODO: Optimization: Can check that stride is a power of 2,
				// and then we could use left shifts instead of multiplication by stride.				
				for (int yy = 0; yy < Height; yy++)
					memcpy_s((byte*)pDst + yy * DstStride, DstStride, (byte*)pSrc + yy * SrcStride, Width * sizeof(PixelType));
			}
		}

		template<typename PixelType> void MoveMemory2D(PixelType* pDst, int DstStride, PixelType* pSrc, int SrcStride, int Width, int Height)
		{
			if (SrcStride == DstStride && SrcStride == Width * sizeof(PixelType))
			{
				memmove_s(pDst, DstStride * Height, pSrc, SrcStride * Height);
			}
			else
			{
				assert(DstStride >= Width * sizeof(PixelType));
				assert(SrcStride >= Width * sizeof(PixelType));

				// TODO: Optimization: Can check that stride is a power of 2,
				// and then we could use left shifts instead of multiplication by stride.				
				for (int yy = 0; yy < Height; yy++)
					memmove_s((byte*)pDst + yy * DstStride, DstStride, (byte*)pSrc + yy * SrcStride, Width * sizeof(PixelType));
			}
		}

		#pragma endregion

		#pragma region "FreeImage compatibility"
		#ifdef FreeImage_Support
		namespace FI
		{
			template<typename PixelType> struct Behaviors
			{
				// Undefined by default- must define specializations or an error will be generated upon attempt to use FreeImage routines.
			};

			template<> struct Behaviors<byte> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_BITMAP, Width, Height, 8); }
			};
			template<> struct Behaviors<UInt16> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_UINT16, Width, Height, 16); }
			};
			template<> struct Behaviors<Int16> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_INT16, Width, Height, 16); }
			};
			template<> struct Behaviors<UInt32> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_UINT32, Width, Height, 32); }
			};
			template<> struct Behaviors<Int32> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_INT32, Width, Height, 32); }
			};
			template<> struct Behaviors<float> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_FLOAT, Width, Height, 32); }
			};
			template<> struct Behaviors<double> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_DOUBLE, Width, Height, 64); }
			};
			template<> struct Behaviors<cuFloatComplex> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_COMPLEX, Width, Height, 64); }
			};
			template<> struct Behaviors<cuDoubleComplex> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_COMPLEX, Width, Height, 128); }
			};
			template<> struct Behaviors<thrust::complex<float>> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_COMPLEX, Width, Height, 64); }
			};
			template<> struct Behaviors<thrust::complex<double>> {
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_COMPLEX, Width, Height, 128); }
			};
			template<> struct Behaviors<RGBPixel> {
				#ifdef LITTLE_ENDIAN
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_BITMAP, Width, Height, 24, 0x000000FF, 0x0000FF00, 0x00FF0000); }
				#else
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_BITMAP, Width, Height, 24, 0x00FF0000, 0x0000FF00, 0x000000FF); }
				#endif
			};
			template<> struct Behaviors<RGBAPixel> {
				#ifdef LITTLE_ENDIAN
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_BITMAP, Width, Height, 32, 0x000000FF, 0x0000FF00, 0x00FF0000); }
				#else
				static FIBITMAP* HostAllocate(int Width, int Height) { return FreeImage_AllocateT(FIT_BITMAP, Width, Height, 32, 0xFF000000, 0x00FF0000, 0x0000FF00); }
				#endif
			};

			inline void ErrorHandler(FREE_IMAGE_FORMAT fif, const char* message) {
				std::string ret = "Image error";
				if (fif != FIF_UNKNOWN) {
					ret += " (" + std::string(FreeImage_GetFormatFromFIF(fif)) + " format): ";
				}
				else ret += ": ";
				ret += std::string(message);
				throw IOException(ret);
			}
		}
		#endif
		#pragma endregion

		#pragma region "Image Memory Management"								

		// Design Notes: HostFlags properly belongs in the wb::images::memory::HostImageData class, but this is too 
		// cumbersome as it gets used often.  Also, embedding it in a class requires that it not be a class enum,
		// as AddFlagSupport() won't work in an embedded setup like that.  Also when I tried it inside the HostImageData
		// class, I had to reference it as:
		//			typedef typename memory::HostImageData<PixelType>::Flags HostFlags;			// Yucky, C++.
		// And one last problem is occasions where the template parameter might not really need to be specified just to
		// give the flags.

		/// <summary>
		/// HostFlags specify rules regarding memory allocation for the host image data.  HostFlags
		/// can be combined together (|).  Pinned must be specified for Portable, Mapped, or WriteCombined.
		/// </summary>		
		enum class HostFlags
		{
			None,				// Allocates ordinary memory.  Not recommended for transfer to device RAM.
			LineAlign,			// Allocates with 16-byte alignment on each scanline.  If not specified, stride is unspecified and depends on libraries.
			Pinned,				// Allocates pinned memory, accessable directly by GPU device.
			Portable,			// Considered pinned for all CUDA contexts, not just the one that performed the allocation.
			Mapped,				// Maps the allocation into the CUDA address space.
			WriteCombined,		// Allocates write-combined (WC), which can be transferred across the PCI Express bus more quickly
								// on some systems, but cannot be read efficiently by most CPUs.  Good for host->device
								// transfers.			

								Retain				// Special flag that can be used in some functions to indicate that flags should be retained as-is.
		};
		AddFlagSupport(HostFlags);		// Enables the use of |, &, ^, |=, &=, and ^= operators on this enumeration.

		namespace memory
		{
			/// <summary>
			/// The DataResponsibility type is used in accounting for who has responsibility for 
			/// freeing image memory, and for what access we are allowed on that memory.  Any memory
			/// allocated by the ImageData classes is going to RdWrR responsibility.  Image data 
			/// that was given via pointer from outside the class may be blocked from certain
			/// operations and would require a copy into a new image before manipulation.
			/// </summary>
			enum class DataResponsibility
			{
				None,				// No data yet, no responsibilities.
				RdWrR,				// Read and write are allowed.  Responsible for memory.
				RdWr,				// Read and write are allowed.  Cannot resize or reallocate memory.
				Rd					// Read-only.
			};

			/// <summary>
			/// The DataState type is used in tracking where the most up-to-date image data is
			/// presently stored.  In some cases, both the host and GPU memory have an up-to-date
			/// copy of the data in which case either can be used as convenient.  As soon as an
			/// operation occurs that modifies the data, that version of the image becomes the
			/// only up-to-date copy.  The memory may still be allocated in both host and device
			/// even if both copies are not up-to-date, and could potentially even have different
			/// dimensions after certain operations.
			/// </summary>
			enum class DataState
			{
				None,
				Host,
				HostAndDevice,
				Device
			};

			/// <summary>
			/// The ImageData class represents one copy of the image data.  It can exist in either
			/// host or device space, depending on the subclass utilized or how the data was
			/// allocated.
			/// </summary>
			/// <typeparam name="PixelType"></typeparam>
			template<typename PixelType> class ImageData
			{
			protected:
				int CalculateAlignedSize(int MinimumSize, int Alignment)
				{
					if (Alignment <= 1) return MinimumSize;
					if (MinimumSize % Alignment == 0) return MinimumSize;
					return (MinimumSize + Alignment - (MinimumSize % Alignment));
					// For example, request a minimum 100 bytes on 128 byte alignment.
					// return (100 + 128 - (100 % 128));
					// return (100 + 128 - 100);
					// return 128;
				}

				/// <summary>
				/// The Destroy() overrides can be called by operator=(), but they won't be reachable by constructors and destructors.  It 
				/// shouldn't actually get called by constructors anyway if they're optimal.  Destructors will need to be virtual and need
				/// to call Destroy().				
				/// </summary>
				virtual void Destroy() { }

			public:

				PixelType*			m_pData;
				int					m_Width;
				int					m_Height;
				int					m_Stride;			// The number of bytes in each line.  Possibly includes unused bytes for alignment.
				DataResponsibility	m_Responsibility;

				ImageData()
				{
					m_pData = nullptr;
					m_Width = m_Height = m_Stride = 0;					
					m_Responsibility = DataResponsibility::None;
				}

				ImageData(ImageData&& mv)
				{
					// Slightly faster than calling operator=(), because operator=() has to call Destroy().
					m_pData = mv.m_pData; mv.m_pData = nullptr;
					m_Width = mv.m_Width; mv.m_Width = 0;
					m_Height = mv.m_Height; mv.m_Height = 0;
					m_Stride = mv.m_Stride; mv.m_Stride = 0;
					m_Responsibility = mv.m_Responsibility;
					mv.m_Responsibility = DataResponsibility::None;
				}

				ImageData<PixelType>& operator=(ImageData&& mv)
				{
					Destroy();
					m_pData = mv.m_pData; mv.m_pData = nullptr;
					m_Width = mv.m_Width; mv.m_Width = 0;
					m_Height = mv.m_Height; mv.m_Height = 0;
					m_Stride = mv.m_Stride; mv.m_Stride = 0;					
					m_Responsibility = mv.m_Responsibility;
					mv.m_Responsibility = DataResponsibility::None;
					return *this;
				}

				virtual ~ImageData() { 
					// Can't call Destroy() here in a useful way, see notes on Destroy().  Descendant classes do need to call it.
				}

				virtual bool IsHostMemory() const = 0;

				/// <summary>
				/// IsShaped() queries whether the requested Width and Height are available
				/// from this image data buffer as-is, and with the necessary write permission.
				/// </summary>				
				/// <returns>True if this ImageData already matches this width, height, and write 
				/// permissions.  False if an Allocate() call would be required to accomodate the 
				/// requested configuration.</returns>
				virtual bool IsShaped(int Width, int Height, bool NeedWritable = false) = 0;

				/// <summary>
				/// AsyncCopyImageTo() transfers the image content from this image into another ImageData.  If 
				/// possible, AsyncCopyImageTo() makes use of the existing image buffer.  If allocation is 
				/// required, then AsyncCopyImageTo() obeys the existing flags and rules established for the 
				/// allocation and does not copy rules to the target ImageData.  For example, if the
				/// target HostImageData object is marked as Pinned memory then AsyncCopyImageTo()
				/// will allocate into Pinned memory even if copying from an unpinned memory object.  
				/// If flags should also be copied from the source object, then the copy constructor
				/// should be used instead.				
				/// </summary>
				/// <param name="dst">An image to receive the copy of data.  Must already have width and height
				/// matching the source image or an exception will be thrown.</param>
				/// <param name="stream">The stream to perform the asynchronous operation on, or the default
				/// stream (0).  The stream is not synchronized and caller is responsible for tracking the
				/// asynchronous operation.</param>								
				void AsyncCopyImageTo(ImageData<PixelType>& dst, cudaStream_t stream)
				{
					// If you want to copy the flags as well as the content, the way to do it is to create a new
					// HostImageData object using the copy constructor.  This is because, in most cases, you can't 
					// change the flags without freeing up the memory and allocating something new.  It's rather
					// messy to allow an unallocated image to float around and be managed by the caller, so the 
					// clean implementation is to require a new object for new flags.  The overhead of the 
					// object itself is small and the overhead of image allocation can't be avoided for 
					// such a case.

					cudaMemcpyKind kind;
					if (IsHostMemory())
					{
						if (dst.IsHostMemory()) kind = cudaMemcpyHostToHost; else kind = cudaMemcpyHostToDevice;
					}
					else
					{
						if (dst.IsHostMemory()) kind = cudaMemcpyDeviceToHost; else kind = cudaMemcpyDeviceToDevice;
					}

					if (!dst.IsShaped(m_Width, m_Height, true)) throw ArgumentException("dst image must already match the dimensions of source image.");					
					cudaThrowable(cudaMemcpy2DAsync(dst.m_pData, dst.m_Stride, m_pData, m_Stride, m_Width * sizeof(PixelType), m_Height, kind, stream));
				}
			};

			template<typename PixelType> class HostImageData : public ImageData<PixelType>
			{
				typedef ImageData<PixelType> base;

				enum { ImageAlignment = 16 /*bytes*/ };
				enum { LineAlignment = 16 /*bytes*/ };

			public:

				// The MSVC compiler will not look in dependent base classes (ImageData<PixelType>) for
				// nondependent names.  These using declarations will resolve the issue.
				using base::m_pData;
				using base::m_Width;
				using base::m_Height;
				using base::m_Stride;
				using base::m_Responsibility;

				#ifdef FreeImage_Support
				/// <summary>
				/// When using FreeImage, host memory that has no flags on will be allocated and freed by FreeImage.  
				/// FreeImage uses the FIBITMAP structure as its data handle.  m_pData will be an alias to 
				/// FreeImage_GetBits().  In cases where FreeImage is responsible for freeing the image memory, 
				/// m_Responsibility will be marked as "not responsible", but the presence of a non-NULL 
				/// m_pFileData ensures responsibility for freeing the handle memory.  This follows the logic that
				/// the m_Responsibility indicator is specifically with regards to m_pData, which in this case
				/// we do not want to free directly.  We instead must free the m_pFileData anytime that it is
				/// non-NULL and treat m_pData as just an alias.
				/// </summary>
				FIBITMAP* m_pFileData;
				#endif

			protected:

				HostFlags	m_Flags;

				bool IsHostMemory() const override { return true; }

				int GetLineAlignment() {
					return ((m_Flags & HostFlags::LineAlign) != 0) ? LineAlignment : 1;
				}

			private:
				void TakeFlags(HostFlags flags)
				{
					m_Flags = flags;
					if ((m_Flags & HostFlags::Portable) != 0
						|| (m_Flags & HostFlags::Mapped) != 0
						|| (m_Flags & HostFlags::WriteCombined) != 0)
					{
						if (!(m_Flags & HostFlags::Pinned))
							throw NotSupportedException("Cannot apply Portable, Mapped, or WriteCombined flags unless Pinned is specified.");
					}
				}

			public:
				HostImageData(HostFlags flags = HostFlags::None)
				#ifdef FreeImage_Support
					: m_pFileData(nullptr)
				#endif
				{
					TakeFlags(flags);
				}				

				#if 0
				HostImageData(HostImageData<PixelType>& cp)
				#ifdef FreeImage_Support
					: m_pFileData(nullptr)
				#endif
				{
					TakeFlags(cp.m_Flags);
					this->m_pData = nullptr;
					cp.AsyncCopyImageTo(*this);
				}
				#endif

				HostImageData(HostImageData<PixelType>&& mv) noexcept : base(std::move(mv))
				{
					TakeFlags(mv.m_Flags);
					#ifdef FreeImage_Support
					m_pFileData = std::move(mv.m_pFileData);
					mv.m_pFileData = nullptr;
					#endif
				}

				HostImageData& operator=(HostImageData<PixelType>&& mv) noexcept
				{
					base::operator=(std::move(mv));
					TakeFlags(mv.m_Flags);
					#ifdef FreeImage_Support
					m_pFileData = std::move(mv.m_pFileData);
					mv.m_pFileData = nullptr;
					#endif
					return *this;
				}

				/// <summary>
				/// Creates a new HostImageData object that wraps another data pointer and can provide an Image object with access to the memory
				/// as if it were an internal Image data buffer.  The data is not copied, and is marked as "not responsible"- that is, the 
				/// caller retains responsibility for ensuring the memory is freed.  The memory cannot be freed until after HostImageData use 
				/// is complete.
				/// </summary>
				/// <param name="Width">Width of the image provided by the pData pointer.</param>
				/// <param name="Height">Height of the image provided by the pData pointer.</param>
				/// <param name="Stride">Stride of the image provided by the pData pointer.</param>
				/// <param name="pData">The image buffer to be wrapped.</param>
				/// <param name="CanWrite">If true, then changes can be made to the underlying data buffer but it cannot be resized.  If false,
				/// then the underlying data buffer is treated as read-only.</param>
				static HostImageData NewWrapper(int Width, int Height, int Stride, PixelType* pData, bool CanWrite, HostFlags flags = HostFlags::None)
				{
					auto ret = HostImageData(flags);
					ret.m_Width = Width;
					ret.m_Height = Height;
					ret.m_Stride = Stride;
					ret.m_pData = pData;
					ret.m_Responsibility = CanWrite ? DataResponsibility::RdWr : DataResponsibility::Rd;
					return ret;
				}

				#ifdef FreeImage_Support
				/// <summary>
				/// Creates a new HostImageData object that takes responsibility for a FreeImage memory buffer and establishes it for use
				/// as an Image data buffer.  The data is not copied but the HostImageData becomes responsible for using FreeImage_Unload()
				/// on the FIBITMAP* when it is no longer needed.
				/// </summary>								
				static HostImageData NewOwner(FIBITMAP* pFIB)
				{
					auto ret = HostImageData(HostFlags::None);
					ret.m_Width = FreeImage_GetWidth(pFIB);
					ret.m_Height = FreeImage_GetHeight(pFIB);
					ret.m_Stride = FreeImage_GetPitch(pFIB);
					ret.m_pData = FreeImage_GetBits(pFIB);
					// See also notes on the m_pFileData member.
					ret.m_Responsibility = DataResponsibility::RdWr;
					ret.m_pFileData = pFIB;
					return ret;
				}
				#endif

				void Destroy() override
				{
					if (this->m_pData != nullptr)
					{
						if (this->m_Responsibility == DataResponsibility::RdWrR)
						{
							if ((m_Flags & HostFlags::Pinned) != 0) {
								#ifdef CUDA_Support
								cudaThrowable(cudaFreeHost(this->m_pData));
								#else
								throw NotSupportedException("Pinned memory requires CUDA support.");
								#endif
							}
							else _aligned_free(this->m_pData);
						}
						this->m_pData = nullptr;
					}
					this->m_Responsibility = DataResponsibility::None;
					this->m_Width = this->m_Height = this->m_Stride = 0;

					#ifdef FreeImage_Support
					if (this->m_pFileData != nullptr) { FreeImage_Unload(this->m_pFileData); this->m_pFileData = nullptr; }
					#endif
				}

				~HostImageData() override {
					Destroy();
				}

				HostFlags GetFlags() const {
					return m_Flags;
				}

				bool IsShaped(int nWidth, int nHeight, bool NeedWritable = false) override
				{
					return
						(m_pData != nullptr && m_Width == nWidth && m_Height == nHeight
							&& (!NeedWritable || m_Responsibility != DataResponsibility::Rd));
				}

				void Allocate(int nWidth, int nHeight, bool NeedWritable = false) 
				{
					/** Can we re-use the memory as-is? **/
					if (IsShaped(nWidth, nHeight, NeedWritable)) return;							// Can use as-is.					

					/** Reallocate or resize? **/
					if (m_pData != nullptr)
					{
						if (m_Responsibility != DataResponsibility::RdWrR)
						{
							// We aren't allowed to touch this pointer.  Since we've been asked to
							// allocate new memory though, we can start new data for which we are
							// responsible.
							m_pData = nullptr;
							m_Width = m_Height = 0;
							m_Responsibility = DataResponsibility::None;

							#ifdef FreeImage_Support
							if (this->m_pFileData != nullptr) { FreeImage_Unload(this->m_pFileData); this->m_pFileData = nullptr; }
							#endif
						}
						else
						{
							if (!(m_Flags & HostFlags::Pinned))
							{
								m_Width = nWidth;
								m_Height = nHeight;
								m_Stride = CalculateAlignedSize(sizeof(PixelType) * nWidth, GetLineAlignment());
								m_pData = (PixelType*)_aligned_realloc(m_pData, m_Stride * m_Height, ImageAlignment);
								// The MS docs on _aligned_realloc() are a bit unclear on error conditions.  It indicates
								// that it can provide an errno, but not exactly when you can consider it an error and
								// therefore when the errno is correctly set by _aligned_realloc() as opposed to a previous
								// function call.  The following is a safe solution, but does not provide any detail to the 
								// error.
								if (m_pData == nullptr) throw OutOfMemoryException();
								return;
							}
							else
							{
								#ifdef CUDA_Support
								if (nWidth <= m_Width && nHeight <= m_Height
									&& (!NeedWritable || m_Responsibility != DataResponsibility::Rd))
								{
									m_Width = nWidth;
									m_Height = nHeight;
									m_Stride = CalculateAlignedSize(sizeof(PixelType) * nWidth, GetLineAlignment());
									return;
								}
								cudaThrowable(cudaFreeHost(m_pData));
								m_pData = nullptr;
								#else
								throw NotSupportedException("Pinned memory requires CUDA support.");
								#endif
							}
						}
					}					

					/** Allocate new memory **/
					m_Width = nWidth;
					m_Height = nHeight;

					#ifdef FreeImage_Support
					if (this->m_pFileData != nullptr) throw Exception("Expected m_pFileData to be null when m_pData is null.");
					if (m_Flags == HostFlags::None)
					{
						assert(m_pData == nullptr);
						FreeImage_SetOutputMessage(FI::ErrorHandler);
						m_pFileData = FI::Behaviors<PixelType>::HostAllocate(m_Width, m_Height);
						if (m_pFileData == nullptr) throw OutOfMemoryException();
						m_Stride = FreeImage_GetPitch(m_pFileData);
						m_pData = (PixelType*)FreeImage_GetBits(m_pFileData);
						if (m_pData == nullptr) throw FormatException();
						m_Responsibility = DataResponsibility::RdWr;
						return;
					}
					#endif

					m_Stride = CalculateAlignedSize(sizeof(PixelType) * nWidth, GetLineAlignment());
					assert(m_pData == nullptr);
					if (!(m_Flags & HostFlags::Pinned))
					{
						m_pData = (PixelType*)_aligned_malloc(m_Stride * m_Height, ImageAlignment);
						if (m_pData == nullptr) wb::Exception::ThrowFromErrno(errno);
					}
					else
					{
						#ifdef CUDA_Support
						unsigned int cudaFlags = 0;
						if (any(m_Flags & HostFlags::Portable)) cudaFlags |= cudaHostAllocPortable;
						if (any(m_Flags & HostFlags::Mapped)) cudaFlags |= cudaHostAllocMapped;
						if (any(m_Flags & HostFlags::WriteCombined)) cudaFlags |= cudaHostAllocWriteCombined;
						cudaThrowable(cudaHostAlloc(&m_pData, m_Stride * m_Height, cudaFlags));
						if (m_pData == nullptr) throw OutOfMemoryException();
						#else
						throw NotSupportedException("Pinned memory requires CUDA support.");
						#endif
					}
					m_Responsibility = DataResponsibility::RdWrR;
				}												
			};

			#ifdef CUDA_Support

			template<typename PixelType> class DeviceImageData : public ImageData<PixelType>
			{
				typedef ImageData<PixelType> base;

				void DoFree()
				{
					assert(m_pData != nullptr);
					cudaThrowable(cudaFree(m_pData));
					m_pData = nullptr;
				}

			protected:

				/// <summary>
				/// The DeviceImageData class allows for "soft" reallocations where the dimensions of the image are changed within a
				/// buffer that is already allocated.  There are certain rules on when this is allowed.  This is particularly helpful
				/// when using a DeviceImageData as a reusable scratch buffer.  In order to facilitate these soft reallocations,
				/// we need to keep track of the true allocated stride and height.
				/// </summary>
				int m_AllocatedStride, m_AllocatedHeight;

				bool IsHostMemory() const override { return false; }

			public:

				// The MSVC compiler will not look in dependent base classes (ImageData<PixelType>) for
				// nondependent names.  These using declarations will resolve the issue.
				using base::m_pData;
				using base::m_Width;
				using base::m_Height;
				using base::m_Stride;
				using base::m_Responsibility;

				bool IsContiguous() const { return m_pData != nullptr && m_Stride == (m_Width * sizeof(PixelType)); }

				bool CanReshape(int NeedWidth, int NeedHeight, bool NeedWritable, bool NeedContiguous)
				{
					return
						(m_pData != nullptr && NeedWidth * sizeof(PixelType) <= m_AllocatedStride 
							&& m_AllocatedHeight <= NeedHeight
							&& (!NeedContiguous || NeedWidth * sizeof(PixelType) == m_AllocatedStride)
							&& (!NeedWritable || m_Responsibility != DataResponsibility::Rd));
				}

				bool IsShaped(int nWidth, int nHeight, bool NeedWritable = false)
				{
					return
						(m_pData != nullptr && m_Width == nWidth && m_Height == nHeight
							&& (!NeedWritable || m_Responsibility != DataResponsibility::Rd));
				}

				bool IsShaped(int nWidth, int nHeight, bool NeedWritable, bool NeedContiguous)
				{
					return
						(m_pData != nullptr && m_Width == nWidth && m_Height == nHeight
							&& (!NeedWritable || m_Responsibility != DataResponsibility::Rd)
							&& (!NeedContiguous || IsContiguous()));
				}

				void Allocate(int NeedWidth, int NeedHeight, bool NeedWritable, bool NeedContiguous, bool& IsPending, cudaStream_t PendingStream, GPUStream& NewStream)
				{
					/** Re-use as-is? **/
					if (IsShaped(NeedWidth, NeedHeight, NeedWritable, NeedContiguous)) return;     // Can use as-is.					

					/** Reallocate, resize, or release? **/
					if (m_pData != nullptr)
					{
						if (m_Responsibility != DataResponsibility::RdWrR)
						{
							// We aren't allowed to touch this pointer.  Since we've been asked to
							// allocate new memory though, we can start new data for which we are
							// responsible.
							m_pData = nullptr;
							m_Width = m_Height = 0;
							m_AllocatedStride = m_AllocatedHeight = 0;
							m_Responsibility = DataResponsibility::None;
						}
						else if (CanReshape(NeedWidth, NeedHeight, NeedWritable, NeedContiguous))
						{
							/** We don't have a cuda reallocation function unfortunately.  For use as scratch
							*   memory, we want to be fast about this, so we essentially have our own
							*   buffer reallocation routine here.
							* 
							*	We don't know exactly what our minimum stride is, but we have probed for it
							*	at GPUSystemInfo construction.
							*/

							m_Width = NeedWidth;
							m_Height = NeedHeight;
							m_Stride = (int)NewStream.GetGSI().GetOptimalPitch(NewStream.GetDeviceId(), (size_t)NeedWidth * sizeof(PixelType));
							if (m_Stride > m_AllocatedStride) throw NotSupportedException("Optimal pitch was identified as larger than previously allocated pitch.");
							// No updates to m_AllocatedStride or m_AllocatedHeight are required.
							return;
						}
						else
						{
						// We can't reallocate in this case.  This is the case where any pending stream
						// operations need to wrap up before we can touch the buffer.
						if (IsPending)
						{
							cudaThrowable(cudaStreamSynchronize(PendingStream));
							IsPending = false;
						}
							DoFree();
						}
					}

					/** Allocate new memory **/
					m_Width = NeedWidth;
					m_Height = NeedHeight;
					m_AllocatedHeight = NeedHeight;					
					if (NeedContiguous)
					{
						m_AllocatedStride = m_Stride = sizeof(PixelType) * NeedWidth;			// Use dense packing for device memory.
						cudaThrowable(cudaMalloc(&m_pData, m_Stride * m_Height));
					}
					else
					{
						// Usually optimal, unless there's a specific reason to stick to contiguous.
						size_t pitch = 0;
						cudaThrowable(cudaMallocPitch(&m_pData, &pitch, sizeof(PixelType) * m_Width, m_Height));
						m_AllocatedStride = m_Stride = (int)pitch;
					}
					if (m_pData == nullptr) throw OutOfMemoryException();
					m_Responsibility = DataResponsibility::RdWrR;
				}

				DeviceImageData() { }

				#if 0
				DeviceImageData(DeviceImageData<PixelType>& cp)
				{
					m_pData = nullptr;
					cp.AsyncCopyImageTo(*this);
				}
				#endif

				DeviceImageData(DeviceImageData<PixelType>&& mv) = default;
				DeviceImageData& operator=(DeviceImageData<PixelType>&& mv) = default;

				void Destroy() override
				{
					if (m_pData != nullptr)
					{
						if (m_Responsibility == DataResponsibility::RdWrR) DoFree();
						m_pData = nullptr;
					}
					m_Responsibility = DataResponsibility::None;
					m_Width = m_Height = m_Stride = 0;
					m_AllocatedHeight = m_AllocatedStride = 0;
				}

				~DeviceImageData() override {
					Destroy();
				}

				cuda::img_format GetCudaFormat() const {
					return cuda::img_format(m_Width, m_Height, m_Stride);
				}
			};			

			#endif	// CUDA_Support

		}// End namespace memory

#pragma endregion

	}// namespace images
}// namespace wb

#endif	// __WBImages_Memory_h__

//	End of Images_Memory.h

