/////////
//	Images_Conversions.h
/////////

#ifndef __WBImages_Conversions_h__
#define __WBImages_Conversions_h__

#include "../../wbCore.h"
#include "../Images.h"

namespace wb
{
	namespace images
	{
		/** Generic / Templates **/
		
		template<typename PixelType, typename FinalType> template<typename PixelTypeDst, typename FinalTypeDst> inline FinalTypeDst& BaseImage<PixelType, FinalType>::TypecastConvertTo(FinalTypeDst& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Destination image must have same size as source image in ConvertTo().");

			if (dst.ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					PixelTypeDst* pDst = dst.GetHostScanlinePtr(yy);
					PixelType* pSrc = GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = static_cast<PixelTypeDst>(*pSrc);
				}
				return dst;
			}
			else
			{
				#ifdef CUDA_Support
				auto formatSrc = m_DeviceData.GetCudaFormat();
				dst.ToDevice();
				auto formatDst = dst.m_DeviceData.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync(m_Stream);
				StartAsync();
				Launch_CUDAKernel(Kernel_TypecastConvertTo, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				AfterKernelLaunch();
				return dst;
				#else
				throw NotSupportedException();
				#endif
			}
		}

		/** From UInt16 **/

		inline Image<float>& Image<UInt16>::ConvertTo(Image<float>& dst) { TypecastConvertTo<float, Image<float>>(dst); return dst; }

		template<> inline Image<float> Image<UInt16>::ConvertTo(HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = GetHostFlags();
			#ifdef CUDA_Support
			auto ret = WouldModifyInHost() ?
				Image<float>::NewHostImage(Width(), Height(), Stream(), Flags)
				: Image<float>::NewDeviceImage(Width(), Height(), Stream(), Flags);
			#else
			auto ret = Image<float>::NewHostImage(Width(), Height(), Stream(), Flags);
			#endif
			ConvertTo(ret);
			return ret;
		}

#if 0
		inline /*static*/ Image<UInt16> Image<UInt16>::CopyFrom(const Image<float>& Src)
		{
			Image<UInt16> ret(Src.Width(), Src.Height());
			for (int yy = 0; yy < Src.Height(); yy++)
				for (int xx = 0; xx < Src.Width(); xx++)
					ret(xx, yy) = (UInt16)Src(xx, yy);
			return ret;
		}

		inline /*static*/ Image<UInt16> Image<UInt16>::CopyFrom(const Image<double>& Src)
		{
			Image<UInt16> ret(Src.Width(), Src.Height());
			for (int yy = 0; yy < Src.Height(); yy++)
				for (int xx = 0; xx < Src.Width(); xx++)
					ret(xx, yy) = (UInt16)Src(xx, yy);
			return ret;
		}

		inline /*static*/ Image<UInt16> Image<UInt16>::CopyFrom(const Image<int>& Src)
		{
			Image<UInt16> ret(Src.Width(), Src.Height());
			for (int yy = 0; yy < Src.Height(); yy++)
				for (int xx = 0; xx < Src.Width(); xx++)
					ret(xx, yy) = (UInt16)Src(xx, yy);
			return ret;
		}

		inline /*static*/ Image<RGBPixel> Image<RGBPixel>::CopyFrom(const Image<byte>& cp)
		{
			Image<RGBPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				byte* pSrcScanline = (byte*)cp.GetScanlinePtr(yy);
				RGBPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					pDstScanline->R = *pSrcScanline;
					pDstScanline->G = *pSrcScanline;
					pDstScanline->B = *pSrcScanline;
				}
			}
			return ret;
		}

		inline /*static*/ Image<RGBPixel> Image<RGBPixel>::CopyFrom(const Image<float>& cp, Image<float>::Range Values)
		{
			float FullRange = Values.Maximum - Values.Minimum;
			Image<RGBPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				float* pSrcScanline = (float*)cp.GetScanlinePtr(yy);
				RGBPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					float Value = 255.0f * (*pSrcScanline - Values.Minimum) / FullRange;
					if (Value < 0.0f) Value = 0.0f; else if (Value > 255.0f) Value = 255.0f;
					pDstScanline->R = (byte)Value;
					pDstScanline->G = (byte)Value;
					pDstScanline->B = (byte)Value;
				}
			}
			return ret;
		}

		inline /*static*/ Image<RGBPixel> Image<RGBPixel>::CopyFrom(const Image<double>& cp, Image<double>::Range Values)
		{
			double FullRange = Values.Maximum - Values.Minimum;
			Image<RGBPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				double* pSrcScanline = (double*)cp.GetScanlinePtr(yy);
				RGBPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					double Value = 255.0 * (*pSrcScanline - Values.Minimum) / FullRange;
					if (Value < 0.0) Value = 0.0; else if (Value > 255.0) Value = 255.0;
					pDstScanline->R = (byte)Value;
					pDstScanline->G = (byte)Value;
					pDstScanline->B = (byte)Value;
				}
			}
			return ret;
		}

		inline /*static*/ Image<RGBAPixel> Image<RGBAPixel>::CopyFrom(const Image<byte>& cp, byte Alpha)
		{
			Image<RGBAPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				byte* pSrcScanline = (byte*)cp.GetScanlinePtr(yy);
				RGBAPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					pDstScanline->R = *pSrcScanline;
					pDstScanline->G = *pSrcScanline;
					pDstScanline->B = *pSrcScanline;
					pDstScanline->A = Alpha;
				}
			}
			return ret;
		}

		inline /*static*/ Image<RGBAPixel> Image<RGBAPixel>::CopyFrom(const Image<float>& cp, Image<float>::Range Values, byte Alpha)
		{
			float FullRange = Values.Maximum - Values.Minimum;
			Image<RGBAPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				float* pSrcScanline = (float*)cp.GetScanlinePtr(yy);
				RGBAPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					float Value = 255.0f * (*pSrcScanline - Values.Minimum) / FullRange;
					if (Value < 0.0f) Value = 0.0f; else if (Value > 255.0f) Value = 255.0f;
					pDstScanline->R = (byte)Value;
					pDstScanline->G = (byte)Value;
					pDstScanline->B = (byte)Value;
					pDstScanline->A = Alpha;
				}
			}
			return ret;
		}

		inline /*static*/ Image<RGBAPixel> Image<RGBAPixel>::CopyFrom(const Image<double>& cp, Image<double>::Range Values, byte Alpha)
		{
			double FullRange = Values.Maximum - Values.Minimum;
			Image<RGBAPixel> ret(cp.Width(), cp.Height());
			for (int yy = 0; yy < cp.Height(); yy++)
			{
				double* pSrcScanline = (double*)cp.GetScanlinePtr(yy);
				RGBAPixel* pDstScanline = ret.GetScanlinePtr(yy);

				for (int xx = 0; xx < cp.Width(); xx++, pSrcScanline++, pDstScanline++)
				{
					double Value = 255.0 * (*pSrcScanline - Values.Minimum) / FullRange;
					if (Value < 0.0) Value = 0.0; else if (Value > 255.0) Value = 255.0;
					pDstScanline->R = (byte)Value;
					pDstScanline->G = (byte)Value;
					pDstScanline->B = (byte)Value;
					pDstScanline->A = Alpha;
				}
			}
			return ret;
		}
#endif

		/** From float **/

		inline Image<byte>& Image<float>::ConvertTo(Image<byte>& dst, Image<float>::Range SrcValueRange)
		{
			float FullRange = SrcValueRange.Maximum - SrcValueRange.Minimum;
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Destination image must have same size as source image in ConvertTo().");

			if (dst.ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					byte* pDst = dst.GetHostScanlinePtr(yy);
					float* pSrc = (float*)GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++)
					{
						float Value = 255.0f * (*pSrc - SrcValueRange.Minimum) / FullRange;
						if (Value < 0.0f) Value = 0.0f; else if (Value > 255.0f) Value = 255.0f;
						*pDst = (byte)(Value);
					}
				}
				return dst;
			}
			else
			{
				throw NotImplementedException("This ConvertTo() has no device-side implementation yet.  Implement or call ToHost() before ConvertTo().");
			}			
		}

		template<> inline Image<byte> Image<float>::ConvertTo(Image<float>::Range SrcValueRange, HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = GetHostFlags();
			#ifdef CUDA_Support
			auto ret = WouldModifyInHost() ?
				Image<byte>::NewHostImage(Width(), Height(), Stream(), Flags)
				: Image<byte>::NewDeviceImage(Width(), Height(), Stream(), Flags);
			#else		
			auto ret = Image<byte>::NewHostImage(Width(), Height(), Stream(), Flags);
			#endif
			ConvertTo(ret, SrcValueRange);
			return ret;
		}

		inline Image<double>& Image<float>::ConvertTo(Image<double>& dst) { TypecastConvertTo<double, Image<double>>(dst); return dst; }

		/** From double **/

		inline Image<float>& Image<double>::ConvertTo(Image<float>& dst) { TypecastConvertTo<float, Image<float>>(dst); return dst; }

		/** From RGBPixel **/

		inline Image<RGBAPixel>& Image<RGBPixel>::ConvertTo(Image<RGBAPixel>& dst, byte Alpha)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Destination image must have same size as source image in ConvertTo().");

			if (dst.ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					RGBPixel* pSrcScanline = (RGBPixel*)GetHostScanlinePtr(yy);
					RGBAPixel* pDstScanline = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrcScanline++, pDstScanline++)
					{
						pDstScanline->R = pSrcScanline->R;
						pDstScanline->G = pSrcScanline->G;
						pDstScanline->B = pSrcScanline->B;
						pDstScanline->A = Alpha;
					}
				}
				return dst;
			}
			else
			{
				throw NotImplementedException("This ConvertTo() has no device-side implementation yet.  Implement or call ToHost() before ConvertTo().");
			}
		}

		template<> inline Image<RGBAPixel> Image<RGBPixel>::ConvertTo(byte Alpha, HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			#ifdef CUDA_Support
			auto ret = WouldModifyInHost() ?
				Image<RGBAPixel>::NewHostImage(Width(), Height(), Stream(), Flags)
				: Image<RGBAPixel>::NewDeviceImage(Width(), Height(), Stream(), Flags);
			#else
			auto ret = Image<RGBAPixel>::NewHostImage(Width(), Height(), Stream(), Flags);
			#endif
			ConvertTo(ret, Alpha);
			return ret;
		}

		template<typename PixelType, typename FinalType>
		template<typename NewPixelType> Image<NewPixelType>& RGBColorImage<PixelType, FinalType>::ConvertToGrayscale(Image<NewPixelType>& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Destination image must have same size as source image in ConvertToGrayscale().");

			if (dst.ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					PixelType* pSrcScanline = (PixelType*)GetHostScanlinePtr(yy);
					NewPixelType* pDstScanline = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrcScanline++, pDstScanline++)
						*pDstScanline = (NewPixelType)(((double)pSrcScanline->R + (double)pSrcScanline->G + (double)pSrcScanline->B) / 3.0);
				}
				return dst;
			}
			else
			{
				throw NotImplementedException("This ConvertTo() has no device-side implementation yet.  Implement or call ToHost() before ConvertTo().");
			}
			return dst;
		}

		template<typename PixelType, typename FinalType>
		template<typename NewPixelType> Image<NewPixelType> RGBColorImage<PixelType, FinalType>::ConvertToGrayscale(HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			#ifdef CUDA_Support
			auto ret = WouldModifyInHost() ?
				Image<NewPixelType>::NewHostImage(Width(), Height(), Stream(), Flags)
				: Image<NewPixelType>::NewDeviceImage(Width(), Height(), Stream(), Flags);
			#else
			auto ret = Image<NewPixelType>::NewHostImage(Width(), Height(), Stream(), Flags);
			#endif
			ConvertToGrayscale(ret);
			return ret;
		}		

	}// namespace images
}// namespace wb

#endif	// __WBImages_Conversions_h__

//	End of Images_Conversions.h

