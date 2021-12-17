/////////
//	Images_Transforms.h
/////////

#ifndef __WBImages_Transforms_h__
#define __WBImages_Transforms_h__

#include "../../wbCore.h"
#include "Images_Base.h"

namespace wb
{
	namespace images
	{
		#ifdef NPP_Support		
		namespace NPPI
		{
			template<typename KernelType> struct TransformBehaviors
			{
				// Undefined by default- must define specializations or an error will be generated upon attempt to use NPPI routines.			
			};
		}
		#endif

		template<typename PixelType, typename FinalType> inline FinalType& BaseImage<PixelType,FinalType>::FlipVerticallyTo(FinalType& dst)
		{
			if (Width() != dst.Width() || Height() != dst.Height())
				throw ArgumentException("Image argument to FlipVerticallyTo() must have the same size as source image.");

			byte* pSrcScanline = nullptr;
			byte* pDstScanline = nullptr;
			cudaMemcpyKind kind = cudaMemcpyHostToHost;
			int nSrcStride, nDstStride;

			if (dst.ModifyInHost(*this))
			{
				nDstStride = dst.m_HostData.m_Stride;
				pDstScanline = (byte*)dst.GetHostScanlinePtr(Height() - 1);

				switch (m_DataState)
				{
				case DataState::Host:
				case DataState::HostAndDevice:
					pSrcScanline = (byte*)GetHostScanlinePtr(0);
					kind = cudaMemcpyHostToHost;
					nSrcStride = m_HostData.m_Stride;
					break;
				case DataState::Device:
					pSrcScanline = (byte*)GetDeviceDataPtr();
					kind = cudaMemcpyDeviceToHost;
					nSrcStride = m_DeviceData.m_Stride;
					break;
				default:
					throw NotImplementedException();
				}
			}
			else
			{
				nDstStride = dst.m_DeviceData.m_Stride;
				int dy = Height() - 1;
				pDstScanline = ((byte*)dst.m_DeviceData.m_pData) + dy * dst.m_DeviceData.m_Stride;

				switch (m_DataState)
				{
				case DataState::Host:
					pSrcScanline = (byte*)GetHostScanlinePtr(0);
					kind = cudaMemcpyHostToHost;
					nSrcStride = m_HostData.m_Stride;
					break;
				case DataState::HostAndDevice:
				case DataState::Device:
					pSrcScanline = (byte*)GetDeviceDataPtr();
					kind = cudaMemcpyDeviceToHost;
					nSrcStride = m_DeviceData.m_Stride;
					break;
				default:
					throw NotImplementedException();
				}
			}

			if (m_Stream.IsNone())
			{
				int MinStride = min(nSrcStride, nDstStride);
				for (int sy = 0; sy < Height(); sy++)
				{
					cudaThrowable(cudaMemcpy(pDstScanline, pSrcScanline, MinStride, kind));

					// Note: it is important that pSrcScanline and pDstScanline be byte* and not PixelType* for the way
					// the following is written (since stride is specified in bytes).
					pSrcScanline += nSrcStride;
					pDstScanline -= nDstStride;
				}
			}
			else
			{
				dst.StartAsync(m_Stream);
				StartAsync();
				int MinStride = min(nSrcStride, nDstStride);
				for (int sy = 0; sy < Height(); sy++)
				{
					cudaThrowable(cudaMemcpyAsync(pDstScanline, pSrcScanline, MinStride, kind, m_Stream));

					// Note: it is important that pSrcScanline and pDstScanline be byte* and not PixelType* for the way
					// the following is written (since stride is specified in bytes).
					pSrcScanline += nSrcStride;
					pDstScanline -= nDstStride;
				}
				// AfterKernelLaunch(); unnecessary, I believe, because cudaMemcpyAsync() returns error codes.								
			}
			
			return dst;
		}

		template<typename PixelType, typename FinalType> inline FinalType BaseImage<PixelType, FinalType>::FlipVerticallyTo(HostFlags Flags) {
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? FinalType::NewHostImage(Width(), Height(), Stream(), Flags) : FinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			FlipVerticallyTo(ret);
			return ret;
		}

		#ifdef NPP_Support		

		template<typename PixelType, typename FinalType> inline FinalType& BaseImage<PixelType, FinalType>::ResizeTo(FinalType& dst, 
			const Rectangle<int>& ToROI, const Rectangle<int>& FromROI,
			InterpolationMethods Method)
		{			
			if (dst.ModifyInHost(*this))
			{
				throw NotImplementedException("Resize is presently only implemented device-side.  Call ToDevice(true) on the dst argument before using.");
			}
			
			ToDevice(false);			
			dst.ToDevice(true);			

			PixelType* pSrc = GetDeviceDataPtr();
			PixelType* pDst = dst.GetDeviceDataPtr();
			int nSrcStride = m_DeviceData.m_Stride, nDstStride = dst.m_DeviceData.m_Stride;

			NPPI::Size SrcSize(Width(), Height());
			NPPI::Rect SrcROI(FromROI.IsWhole() ? Bounds() : FromROI);
			NPPI::Size DstSize(dst.Width(), dst.Height());
			NPPI::Rect DstROI(ToROI.IsWhole() ? dst.Bounds() : ToROI);
						
			dst.StartAsync(m_Stream);
			StartAsync();
			cudaThrowable(
				NPPI::TransformBehaviors<PixelType>::nppiResizeROI(
					pSrc, nSrcStride,
					SrcSize, SrcROI,
					pDst, nDstStride,
					DstSize, DstROI,
					(int)Method,
					m_Stream
				));
			return dst;
		}

		template<typename PixelType, typename FinalType> inline FinalType BaseImage<PixelType, FinalType>::ResizeTo(const Rectangle<int>& ToROI, 
			const Rectangle<int>& FromROI, InterpolationMethods Method,
			HostFlags Flags)
		{
			// Note: ToROI can't be a default argument in this overload like in the other because it defines the image size to be created.
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ?
				FinalType::NewHostImage(ToROI.X + ToROI.Width, ToROI.Y + ToROI.Height, Stream(), Flags)
				: FinalType::NewDeviceImage(ToROI.X + ToROI.Width, ToROI.Y + ToROI.Height, Stream(), Flags);
			ResizeTo(ret, ToROI, FromROI, Method);
			return ret;
		}

		#endif
				
		template<typename PixelType, typename FinalType> inline FinalType& BaseImage<PixelType, FinalType>::CopyTo(FinalType& dst, Rectangle<int> FromROI, int ToX, int ToY)
		{
			if (FromROI.IsWhole()) FromROI = Bounds();
			if (ToX + FromROI.Width > dst.Width() || ToY + FromROI.Height > dst.Height()) 
				throw ArgumentException("Destination image provided to CopyTo() must be larger enough to accomodate the ROI and offset specified.");

			byte* pSrc = nullptr;
			byte* pDst = nullptr;
			cudaMemcpyKind kind = cudaMemcpyHostToHost;
			int nSrcStride, nDstStride;
			bool ModifyingHost = dst.ModifyInHost(*this);

			if (ModifyingHost)
			{
				nDstStride = dst.m_HostData.m_Stride;
				pDst = (byte*)dst.GetHostScanlinePtr(0);

				switch (m_DataState)
				{
				case DataState::Host:
				case DataState::HostAndDevice:
					Synchronize();
					pSrc = (byte*)GetHostScanlinePtr(0);
					kind = cudaMemcpyHostToHost;
					nSrcStride = m_HostData.m_Stride;
					break;
				case DataState::Device:
					pSrc = (byte*)GetDeviceDataPtr();
					kind = cudaMemcpyDeviceToHost;
					nSrcStride = m_DeviceData.m_Stride;
					break;
				default:
					throw NotImplementedException();
				}
			}
			else
			{
				nDstStride = dst.m_DeviceData.m_Stride;
				pDst = (byte*)dst.GetDeviceDataPtr();

				switch (m_DataState)
				{
				case DataState::Host:
					Synchronize();
					pSrc = (byte*)GetHostScanlinePtr(0);
					kind = cudaMemcpyHostToHost;
					nSrcStride = m_HostData.m_Stride;
					break;
				case DataState::HostAndDevice:
				case DataState::Device:
					pSrc = (byte*)GetDeviceDataPtr();
					kind = cudaMemcpyDeviceToHost;
					nSrcStride = m_DeviceData.m_Stride;
					break;
				default:
					throw NotImplementedException();
				}
			}

			pSrc += FromROI.X * sizeof(PixelType);
			pSrc += FromROI.Y * nSrcStride;
			pDst += ToX * sizeof(PixelType);
			pDst += ToY * nDstStride;

			int WidthInBytes = FromROI.Width * sizeof(PixelType);
			if (m_Stream.IsNone())
			{								
				cudaThrowable(cudaMemcpy2D(pDst, nDstStride, pSrc, nSrcStride, WidthInBytes, FromROI.Height, kind));
			}
			else
			{
				dst.StartAsync(m_Stream);
				StartAsync(m_Stream);
				cudaThrowable(cudaMemcpy2DAsync(pDst, nDstStride, pSrc, nSrcStride, WidthInBytes, FromROI.Height, kind, m_Stream));
				// AfterKernelLaunch(); unnecessary, I believe, because cudaMemcpyAsync() returns error codes.								
			}

			//if (ModifyingHost)
			{
				// This isn't really optimal if the caller has other work they can do while waiting for the
				// stream to finish asynchronously.  This prevents any host access before the data has arrived though.
				//dst.Synchronize();
			}
			return dst;
		}						

		template<typename PixelType, typename FinalType> inline FinalType BaseImage<PixelType, FinalType>::CropTo(const Rectangle<int>& FromROI, HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? FinalType::NewHostImage(FromROI.Width, FromROI.Height, Stream(), Flags) : FinalType::NewDeviceImage(FromROI.Width, FromROI.Height, Stream(), Flags);
			CopyTo(ret, FromROI, 0, 0);
			return ret;
		}

		template<typename PixelType, typename FinalType> inline FinalType BaseImage<PixelType, FinalType>::CropTo(int x1, int y1, int Width, int Height, HostFlags Flags)
		{
			return CropTo(Rectangle<int>(x1, y1, Width, Height), Flags);
		}

		#pragma region "NPPI Multiplexors"

		namespace NPPI
		{
			template<> struct TransformBehaviors<byte>
			{
				static NppStatus nppiResizeROI(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_8u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};

			template<> struct TransformBehaviors<UInt16>
			{
				static NppStatus nppiResizeROI(const Npp16u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16u* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_16u_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};

			template<> struct TransformBehaviors<Int16>
			{
				static NppStatus nppiResizeROI(const Npp16s* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp16s* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_16s_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};
			
			template<> struct TransformBehaviors<float>
			{
				static NppStatus nppiResizeROI(const Npp32f* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp32f* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_32f_C1R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};

			template<> struct TransformBehaviors<RGBPixel>
			{
				static NppStatus nppiResizeROI(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_8u_C3R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};

			template<> struct TransformBehaviors<RGBAPixel>
			{
				static NppStatus nppiResizeROI(const Npp8u* pSrc, int nSrcStep, NppiSize oSrcSize, NppiRect oSrcRectROI, Npp8u* pDst, int nDstStep, NppiSize oDstSize, NppiRect oDstRectROI, int eInterpolation, NppStreamContext nppStreamCtx) {
					return nppiResize_8u_C4R_Ctx(pSrc, nSrcStep, oSrcSize, oSrcRectROI, pDst, nDstStep, oDstSize, oDstRectROI, eInterpolation, nppStreamCtx);
				}
			};
		}

		#pragma endregion

	}// namespace images
}// namespace wb

#endif	// __WBImages_Transforms_h__

//	End of Images_Transforms.h

