/////////
//	Images_Convolution.h
/////////

#ifndef __WBImages_Convolution_h__
#define __WBImages_Convolution_h__

#include "../../wbCore.h"
#include "Images_Base.h"

namespace wb
{
	namespace images
	{
		/** References **/

		using namespace wb::cuda;

		#pragma region "Convolution Kernels"		

		#ifdef NPP_Support		

		/// <summary>Defines a kernel to be used for image convolution.  The easiest way to specify a ConvolutionKernel is with 
		/// the New() routine, i.e.:
		/// <code>
		///		auto LPF = ConvolutionKernel&lt;float&gt;::New(3, 3, 
		///					{1.0f/9, 1.0f/9, 1.0f/9,   1.0f/9, 1.0f/9, 1.0f/9,   1.0f/9, 1.0f/9, 1.0f/9});
		/// </code>
		/// (Note, the above requires a modern C++ compiler, but you could initialize the vector before the call in an older
		/// compiler.)
		/// 
		/// The Kernel uses the reversed order, same as MATLAB.
		/// 
		/// The New() routine make the kernel available on both the host and device side at construction, so that the kernel
		/// object can be saved for later use and already be ready.  However, if this is not the desired behavior, the
		/// ConvolutionKernel&lt;PixelType&gt;::NewHostImage() or NewDeviceImage() routines can be used as long as the
		/// LineAligned flag is absent.
		///	</summary>		
		template<typename PixelType> class ConvolutionKernel : public BaseImage<PixelType, ConvolutionKernel<PixelType>>
		{
			typedef BaseImage<PixelType, ConvolutionKernel<PixelType>> base;

		protected:
			ConvolutionKernel(HostFlags HostFlags, GPUStream Stream) : base(HostFlags, Stream) { }
		public:
			ConvolutionKernel() = default;
			ConvolutionKernel(ConvolutionKernel&&) = default;
			ConvolutionKernel(ConvolutionKernel&) = delete;
			ConvolutionKernel& operator=(ConvolutionKernel&&) = default;
			ConvolutionKernel& operator=(ConvolutionKernel&) = delete;

			PixelType Divisor;

			void ToDevice(bool Writable = false) override
			{
				m_TowardHost = false;
				if (m_DataState == DataState::HostAndDevice && !m_DeviceData.IsContiguous()) m_DataState = DataState::Host;
				switch (m_DataState)
				{
				case DataState::Host:
					m_DeviceData.Allocate(m_HostData.m_Width, m_HostData.m_Height, Writable, true);
					m_DeviceData.CopyImageFrom(m_HostData);
					if (Writable) m_DataState = DataState::Device; else m_DataState = DataState::HostAndDevice;
					return;
				case DataState::HostAndDevice:
					if (Writable) m_DataState = DataState::Device;
				case DataState::Device:
					if (!m_DeviceData.IsContiguous()) throw NotSupportedException("Device data must be contiguously allocated for a convolution kernel.");
					return;
				default: throw NotSupportedException();
				}
			}

			static ConvolutionKernel<PixelType> New(int nWidth, int nHeight, vector<PixelType> Kernel, GPUStream Stream, PixelType Divisor = 1)
			{
				throw NotImplementedException("TODO: This all works, but I want to consider applying an order-swapping so that it isn't counterintuitive what the kernel looks like.");

				// Note: it is crucial that the HostFlags::LineAlign flag be absent from a convolution kernel for the NPP library.  
				// The NPP library expects kernels to be contiguous memory.
				ConvolutionKernel<PixelType> ret = ConvolutionKernel<PixelType>::NewHostImage(nWidth, nHeight, Stream, HostFlags::Pinned);
				PixelType* pDst = ret.GetHostScanlinePtr(0);
				MoveMemory(pDst, Kernel.data(), sizeof(PixelType) * nWidth * nHeight);							

				if (NPPI::KernelBehaviors<PixelType>::ShouldNormalizeKernelUpFront()) { ret /= Divisor; ret.Divisor = 1; }
				else ret.Divisor = Divisor;

				// Go ahead and ensure the kernel is available in the device as well.  The reasoning here is that in an optimized code you
				// would usually allocate your convolution kernels statically/globally/once and then re-use them.  So we can do this at
				// initialization and have it ready.  If this is not the desired behavior, then the caller can use NewHostImage() or
				// NewDeviceImage() instead.
				ret.ToDevice(false);

				return ret;
			}

			/// <summary>
			/// Initializes a new ConvolutionKernel object from a given kernel, which can be specified as a literal in C++11.  The specification
			/// is row-wise, such that New({ {1, 1, 1}, {2, 3, 4} }) will yield an image width width of 3 and height of 2.
			/// </summary>
			static ConvolutionKernel<PixelType> New(vector<vector<PixelType>> Kernel, GPUStream Stream, PixelType Divisor = 1)
			{
				throw NotImplementedException("TODO: This all works, but I want to consider applying an order-swapping so that it isn't counterintuitive what the kernel looks like.");

				int nHeight = (int)Kernel.size();
				if (nHeight < 1) throw ArgumentException("Kernel argument cannot be an empty vector.");
				int nWidth = (int)Kernel[0].size();

				// Note: it is crucial that the HostFlags::LineAlign flag be absent from a convolution kernel for the NPP library.  
				// The NPP library expects kernels to be contiguous memory.
				ConvolutionKernel<PixelType> ret = ConvolutionKernel<PixelType>::NewHostImage(nWidth, nHeight, Stream, HostFlags::Pinned);
				for (int yy = 0; yy < nHeight; yy++)
				{
					if (Kernel[yy].size() != nWidth) throw ArgumentException("Kernel argument must provide a uniform matrix, not jagged.");

					PixelType* pDst = ret.GetHostScanlinePtr(yy);
					PixelType* pSrc = Kernel[yy].data();
					MoveMemory(pDst, pSrc, sizeof(PixelType) * nWidth);
				}

				if (NPPI::KernelBehaviors<PixelType>::ShouldNormalizeKernelUpFront()) { ret /= Divisor; ret.Divisor = 1; }
				else ret.Divisor = Divisor;

				// Go ahead and ensure the kernel is available in the device as well.  The reasoning here is that in an optimized code you
				// would usually allocate your convolution kernels statically/globally/once and then re-use them.  So we can do this at
				// initialization and have it ready.  If this is not the desired behavior, then the caller can use NewHostImage() or
				// NewDeviceImage() instead.
				ret.ToDevice(false);

				return ret;
			}
		};		

		#endif // NPP_Support

		#pragma endregion

		#pragma region "Convolution Kernels - NPP Multiplexers"
		#ifdef NPP_Support

		namespace NPPI
		{
			template<> struct KernelBehaviors<Int32>
			{
				// This function refers to the type of the kernel and not of the image:
				static bool ShouldNormalizeKernelUpFront() { return false; }
			};

			template<> struct KernelBehaviors<float>
			{
				// This function refers to the type of the kernel and not of the image:
				static bool ShouldNormalizeKernelUpFront() { return true; }
			};

			template<> struct KernelBehaviors<double>
			{
				// This function refers to the type of the kernel and not of the image:
				static bool ShouldNormalizeKernelUpFront() { return true; }
			};
		}

		namespace NPPI
		{
			template<> struct FilterBehaviors<byte, Int32>
			{
				static NppStatus nppiFilter(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
					return nppiFilter_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
				}
				static NppStatus nppiFilterBorder(const Npp8u* pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, eBorderType, nppStreamCtx);
				}
				/*
				static NppStatus nppiFilterBoxBorder(const Npp8u* pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI, NppiSize oMaskSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBoxBorder_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, oMaskSize, oAnchor, eBorderType, nppStreamCtx);
				}
				*/
			};

			template<> struct FilterBehaviors<byte, float>
			{
				static NppStatus nppiFilter(const Npp8u* pSrc, int nSrcStep, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
					return nppiFilter32f_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
				}
				static NppStatus nppiFilterBorder(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp8u* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBorder32f_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, eBorderType, nppStreamCtx);
				}
			};

			template<> struct FilterBehaviors<UInt16, Int32>
			{
				static NppStatus nppiFilter(const Npp16u* pSrc, Npp32s nSrcStep, Npp16u* pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppStreamContext nppStreamCtx) {
					return nppiFilter_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, nppStreamCtx);
				}
				static NppStatus nppiFilterBorder(const Npp16u* pSrc, Npp32s nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u* pDst, Npp32s nDstStep, NppiSize oSizeROI, const Npp32s* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, Npp32s nDivisor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBorder_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nDivisor, eBorderType, nppStreamCtx);
				}
			};
			template<> struct FilterBehaviors<UInt16, float>
			{
				static NppStatus nppiFilter(const Npp16u* pSrc, int nSrcStep, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
					return nppiFilter32f_16u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
				}
				static NppStatus nppiFilterBorder(const Npp16u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp16u* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBorder32f_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, eBorderType, nppStreamCtx);
				}
			};
		
			template<> struct FilterBehaviors<float, float>
			{
				static NppStatus nppiFilter(const Npp32f* pSrc, int nSrcStep, Npp32f* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
					return nppiFilter_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
				}
				static NppStatus nppiFilterBorder(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSize, NppiPoint oSrcOffset, Npp32f* pDst, int nDstStep, NppiSize oSizeROI, const Npp32f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppiBorderType eBorderType, NppStreamContext nppStreamCtx) {
					return nppiFilterBorder_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcOffset, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, eBorderType, nppStreamCtx);
				}
			};
			template<> struct FilterBehaviors<double, double>
			{
				static NppStatus nppiFilter(const Npp64f* pSrc, int nSrcStep, Npp64f* pDst, int nDstStep, NppiSize oSizeROI, const Npp64f* pKernel, NppiSize oKernelSize, NppiPoint oAnchor, NppStreamContext nppStreamCtx) {
					return nppiFilter_64f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, pKernel, oKernelSize, oAnchor, nppStreamCtx);
				}
				// No Border version provided in nppi for Image<double>.
			};
		}		

		#endif	// NPP_Support
		#pragma endregion

		#pragma region "Convolution Implementation"
		#ifdef NPP_Support

		template<typename PixelType, typename FinalType> template<typename KernelType> inline FinalType& BaseImage<PixelType,FinalType>::Convolve(FinalType& Filtered, 
			const ConvolutionKernel<KernelType>& Kernel, const Rectangle<int>& ROI)
		{
			if (!ROI.IsWhole() && !ROI.IsContainedIn(Bounds())) throw ArgumentException("Requested ROI does not fit within image bounds.");

			if (Filtered.ModifyInHost())
			{
				throw NotImplementedException("Convolution is presently only implemented device-side.  Call ToDevice(true) on the Filtered argument before using.");
			}

			// Only implemented device-side at present.
			ToDevice(false);
			Filtered.ToDevice(true);
			if (!Kernel.IsReadableOnDevice()) throw ArgumentException("Kernel must be readable on device.");
			if (!Kernel.m_DeviceData.IsContiguous()) throw ArgumentException("Kernel must be in contiguous memory on device.");		// Inaccessible, but a good check.

			byte* pSrcTL = (((byte*)GetDeviceDataPtr()) + (ROI.Y * m_DeviceData.m_Stride) + (ROI.X * sizeof(PixelType)));
			byte* pDstTL = (((byte*)Filtered.GetDeviceDataPtr()) + (ROI.Y * Filtered.m_DeviceData.m_Stride) + (ROI.X * sizeof(PixelType)));
			int nSrcStride = m_DeviceData.m_Stride, nDstStride = Filtered.m_DeviceData.m_Stride;

			NPPI::Size oSrcSize(Width(), Height());
			NPPI::Point oSrcOffset(ROI.X, ROI.Y);

			auto oSizeROI = ROI.IsWhole() ? NPPI::Size(Width(), Height()) : NPPI::Size(ROI.Width, ROI.Height);

			NPPI::Size oKernelSize(Kernel.Width(), Kernel.Height());
			NPPI::Point oAnchor(Kernel.Width() / 2, Kernel.Height() / 2);

			Filtered.StartAsync(m_Stream);
			StartAsync();
			if (ROI.IsWhole())
				cudaThrowable(
					NPPI::FilterBehaviors<PixelType COMMA KernelType>::nppiFilter(
						/*Input Image=*/ (const Npp8u*)pSrcTL, nSrcStride,
						/*Output Image=*/ (Npp8u*)pDstTL, nDstStride,
						/*Output ROI Size=*/ oSizeROI,
						/*Kernel=*/ (const Npp32s*)Kernel.GetDeviceDataPtr(), oKernelSize, oAnchor,
						/*Misc=*/ Kernel.Divisor,
						m_Stream
					));
			else
				cudaThrowable(
					NPPI::FilterBehaviors<PixelType COMMA KernelType>::nppiFilterBorder(
						/*Input Image=*/ (const Npp8u*)pSrcTL, nSrcStride, oSrcSize, oSrcOffset,
						/*Output Image=*/ (Npp8u*)pDstTL, nDstStride,
						/*Output ROI Size=*/ oSizeROI,
						/*Kernel=*/ (const Npp32s*)Kernel.GetDeviceDataPtr(), oKernelSize, oAnchor,
						/*Misc=*/ Kernel.Divisor, NPP_BORDER_REPLICATE,
						m_Stream
					));
			return Filtered;
		}

		/// <summary>
		/// Applies a general convolution kernel to an image and returns a new image.
		/// Note: Not all KernelTypes are supported.  Int32 and float are generally supported,
		/// and double is supported for double images.
		/// </summary>			
		/// <param name="Kernel">The general convolution kernel to apply.</param>			
		/// <returns></returns>
		template<typename PixelType, typename FinalType> template<typename KernelType> inline FinalType BaseImage<PixelType, FinalType>::Convolve(const ConvolutionKernel<KernelType>& Kernel, const Rectangle<int>& ROI)
		{
			auto ret = FinalType::NewDeviceImage(Width(), Height(), GetHostFlags(), Stream());
			Convolve(ret, Kernel, ROI);
			return ret;
		}

		#endif
		#pragma endregion

	}// namespace images
}// namespace wb

#endif	// __WBImages_Convolution_h__

//	End of Images_Convolution.h

