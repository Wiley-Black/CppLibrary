/////////
//	Images_Math.h
/////////

#ifndef __WBImages_Math_h__
#define __WBImages_Math_h__

#ifndef __WBImages_h__
#error Include this header only via Images.h.
#endif

#include "../Images.h"

#include <thrust/complex.h>

namespace wb
{
	namespace images
	{		
		#pragma region "Specialized complex images"

		template<>
		class Image<thrust::complex<float>> : public ComplexImage<thrust::complex<float>, float, Image<thrust::complex<float>>, Image<float>>
		{
			typedef float RealPixelType;
			typedef thrust::complex<RealPixelType> PixelType;			
			typedef ComplexImage<PixelType, RealPixelType, Image<PixelType>, Image<RealPixelType>> base;
			friend BaseImage<PixelType, Image<PixelType>>;
			friend base;

		protected:
			Image(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			Image() = default;
			Image(Image&&) = default;
			Image(Image&) = delete;
			Image& operator=(Image&&) = default;
			Image& operator=(Image&) = delete;

			static Image Load(const std::string& filename, GPUStream Stream = GPUStream::None());
			void Save(const std::string& filename);
		};

		template<>
		class Image<thrust::complex<double>> : public ComplexImage<thrust::complex<double>, double, Image<thrust::complex<double>>, Image<double>>
		{
			typedef double RealPixelType;
			typedef thrust::complex<RealPixelType> PixelType;
			typedef ComplexImage<PixelType, RealPixelType, Image<PixelType>, Image<RealPixelType>> base;
			friend BaseImage<PixelType, Image<PixelType>>;
			friend base;

		protected:
			Image(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			Image() = default;
			Image(Image&&) = default;
			Image(Image&) = delete;
			Image& operator=(Image&&) = default;
			Image& operator=(Image&) = delete;

			static Image Load(const std::string& filename, GPUStream Stream = GPUStream::None());
			void Save(const std::string& filename);
		};

		#pragma endregion

		#pragma region "NPPI Support"
		#ifdef NPP_Support

		namespace NPPI
		{
			template<> struct Behaviors<byte>
			{
				static NppStatus SumReal(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pSum, NppStreamContext nppStreamCtx)
				{
					return nppiSum_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pSum, nppStreamCtx);
				}
				static NppStatus MinReal(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp8u* pMin, NppStreamContext nppStreamCtx)
				{
					return nppiMin_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMin, nppStreamCtx);
				}
				static NppStatus MaxReal(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp8u* pMax, NppStreamContext nppStreamCtx)
				{
					return nppiMax_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMax, nppStreamCtx);
				}
				static NppStatus MeanAndStdDevReal(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pMean, Npp64f* pStdDev, NppStreamContext nppStreamCtx)
				{
					return nppiMean_StdDev_8u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx);
				}
				static NppStatus GetSumRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiSumGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMinRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMinGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMaxRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMaxGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMeanAndStdDevRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus FilterSobelHoriz(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
				{
					return nppiFilterSobelHoriz_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
				}
				static NppStatus FilterSobelVert(const Npp8u* pSrc, Npp32s nSrcStep, Npp8u* pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
				{
					return nppiFilterSobelVert_8u_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
				}
			};
			template<> struct Behaviors<UInt16>
			{
				static NppStatus SumReal(const Npp16u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pSum, NppStreamContext nppStreamCtx)
				{
					return nppiSum_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pSum, nppStreamCtx);
				}
				static NppStatus MinReal(const Npp16u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp16u* pMin, NppStreamContext nppStreamCtx)
				{
					return nppiMin_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMin, nppStreamCtx);
				}
				static NppStatus MaxReal(const Npp16u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp16u* pMax, NppStreamContext nppStreamCtx)
				{
					return nppiMax_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMax, nppStreamCtx);
				}
				static NppStatus MeanAndStdDevReal(const Npp16u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pMean, Npp64f* pStdDev, NppStreamContext nppStreamCtx)
				{
					return nppiMean_StdDev_16u_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx);
				}
				static NppStatus GetSumRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiSumGetBufferHostSize_16u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMinRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMinGetBufferHostSize_16u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMaxRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMaxGetBufferHostSize_16u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMeanAndStdDevRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMeanStdDevGetBufferHostSize_16u_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
			};
			template<> struct Behaviors<float>
			{
				static NppStatus SumReal(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pSum, NppStreamContext nppStreamCtx)
				{
					return nppiSum_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pSum, nppStreamCtx);
				}
				static NppStatus MinReal(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp32f* pMin, NppStreamContext nppStreamCtx)
				{
					return nppiMin_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMin, nppStreamCtx);
				}
				static NppStatus MaxReal(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp32f* pMax, NppStreamContext nppStreamCtx)
				{
					return nppiMax_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMax, nppStreamCtx);
				}
				static NppStatus MeanAndStdDevReal(const Npp32f* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pMean, Npp64f* pStdDev, NppStreamContext nppStreamCtx)
				{
					return nppiMean_StdDev_32f_C1R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean, pStdDev, nppStreamCtx);
				}
				static NppStatus GetSumRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiSumGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMinRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMinGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMaxRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMaxGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus GetMeanAndStdDevRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
				static NppStatus FilterSobelHoriz(const Npp32f* pSrc, Npp32s nSrcStep, Npp32f* pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
				{
					return nppiFilterSobelHoriz_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
				}
				static NppStatus FilterSobelVert(const Npp32f* pSrc, Npp32s nSrcStep, Npp32f* pDst, Npp32s nDstStep, NppiSize oSizeROI, NppStreamContext nppStreamCtx)
				{
					return nppiFilterSobelVert_32f_C1R_Ctx(pSrc, nSrcStep, pDst, nDstStep, oSizeROI, nppStreamCtx);
				}
			};
			template<> struct Behaviors<RGBPixel>
			{
				static NppStatus SumRGB(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pSum3, NppStreamContext nppStreamCtx)
				{
					return nppiSum_8u_C3R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pSum3, nppStreamCtx);
				}
				static NppStatus GetSumRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiSumGetBufferHostSize_8u_C3R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
			};
			template<> struct Behaviors<RGBAPixel>
			{
				static NppStatus SumRGB(const Npp8u* pSrc, int nSrcStep, NppiSize oSizeROI, Npp8u* pDeviceBuffer, Npp64f* pSum3, NppStreamContext nppStreamCtx)
				{
					return nppiSum_8u_AC4R_Ctx(pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pSum3, nppStreamCtx);
				}
				static NppStatus GetSumRealBufferSize(NppiSize oSizeROI, int* hpBufferSize, NppStreamContext nppStreamCtx)
				{
					return nppiSumGetBufferHostSize_8u_AC4R_Ctx(oSizeROI, hpBufferSize, nppStreamCtx);
				}
			};
			template<> struct Behaviors<thrust::complex<float>>
			{
				static NppStatus SumComplex(const Npp32fc* pSrc, int nLength, Npp32fc* pSum, Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx)
				{
					return nppsSum_32fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
				}
				static NppStatus GetSumComplexBufferSize(int nLength, int* hpBufferSize) {
					return nppsSumGetBufferSize_32fc(nLength, hpBufferSize);
				}
			};
			template<> struct Behaviors<thrust::complex<double>>
			{
				static NppStatus SumComplex(const Npp64fc* pSrc, int nLength, Npp64fc* pSum, Npp8u* pDeviceBuffer, NppStreamContext nppStreamCtx)
				{
					return nppsSum_64fc_Ctx(pSrc, nLength, pSum, pDeviceBuffer, nppStreamCtx);
				}
				static NppStatus GetSumComplexBufferSize(int nLength, int* hpBufferSize) {
					return nppsSumGetBufferSize_64fc(nLength, hpBufferSize);
				}
			};
		}

		#endif
		#pragma endregion

		#pragma region "Complex<->Real"

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline FinalType& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::SetReal(RealImage<RealPixelType, RealFinalType>& src)
		{
			if (src.Width() != Width() || src.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (ModifyInHost(src))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = src.GetHostScanlinePtr(yy);
					auto pDst = GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = thrust::complex<RealPixelType>(*pSrc, pDst->imag());
				}
			}
			else
			{
				auto formatSrc = src.GetCudaFormat();
				auto formatDst = m_DeviceData.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				src.ToDevice(false);
				ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				StartAsync();
				src.StartAsync(Stream());
				Launch_CUDAKernel(Kernel_ComplexSetReal, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					GetDeviceDataPtr(), formatDst,
					src.GetDeviceDataPtr(), formatSrc
				);
				AfterKernelLaunch();
			}
			return (FinalType&)*this;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline FinalType& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::SetImag(RealImage<RealPixelType, RealFinalType>& src)
		{
			if (src.Width() != Width() || src.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (ModifyInHost(src))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = src.GetHostScanlinePtr(yy);
					auto pDst = GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = thrust::complex<RealPixelType>(pDst->real(), *pSrc);
				}
			}
			else
			{
				auto formatSrc = src.m_DeviceData.GetCudaFormat();
				auto formatDst = m_DeviceData.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				src.ToDevice(false);
				ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				StartAsync();
				src.StartAsync(Stream());
				Launch_CUDAKernel(Kernel_ComplexSetImag, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					GetDeviceDataPtr(), formatDst,
					src.GetDeviceDataPtr(), formatSrc
				);
				AfterKernelLaunch();
			}
			return (FinalType&)*this;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealImage<RealPixelType, RealFinalType>& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::GetRealTo(RealImage<RealPixelType, RealFinalType>& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (ModifyInHost(dst))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = pSrc->real();
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync(Stream());
				StartAsync();
				Launch_CUDAKernel(Kernel_ComplexGetReal, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealFinalType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::GetReal(HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? RealFinalType::NewHostImage(Width(), Height(), Stream(), Flags) : RealFinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			GetRealTo(ret);
			return ret;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealImage<RealPixelType, RealFinalType>& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::GetImagTo(RealImage<RealPixelType, RealFinalType>& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (ModifyInHost(dst))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = pSrc->imag();
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync(Stream());
				StartAsync();
				Launch_CUDAKernel(Kernel_ComplexGetImag, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealFinalType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::GetImag(HostFlags Flags)
		{
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? RealFinalType::NewHostImage(Width(), Height(), Stream(), Flags) : RealFinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			GetImagTo(ret);
			return ret;
		}

		#pragma endregion

		#pragma region "Complex Conjugation"

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType> 
		inline ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::ConjugateTo(ComplexImage& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to ConjugateTo() to have same size.");

			if (dst.ModifyInHost(*this))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = thrust::conj(*pSrc);
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.m_DeviceData.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				dst.GetSmallOpKernelParameters(blocks, threads);
				
				dst.StartAsync();
				StartAsync(dst.Stream());
				Launch_CUDAKernel(Kernel_ComplexConjugate, blocks, threads, /*dynamic shared memory=*/ 0, dst.m_Stream,
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				dst.AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline FinalType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::Conjugate(HostFlags Flags) {
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? FinalType::NewHostImage(Width(), Height(), Stream(), Flags) : FinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			ConjugateTo(ret);
			return ret;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline FinalType& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::ConjugateInPlace()
		{			
			if (ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pImg = GetHostScanlinePtr(yy);					

					for (int xx = 0; xx < Width(); xx++, pImg++) *pImg = thrust::conj(*pImg);
				}
			}
			else
			{				
				auto format = m_DeviceData.GetCudaFormat();								
				ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				StartAsync();
				Launch_CUDAKernel(Kernel_ComplexConjugateInPlace, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					GetDeviceDataPtr(), format
				);
				AfterKernelLaunch();
			}
			return (FinalType&)*this;
		}

		#pragma endregion

		#pragma region "Absolute Value, Real"

		template<typename PixelType, typename FinalType> inline RealImage<PixelType, FinalType>& RealImage<PixelType, FinalType>::AbsoluteTo(RealImage& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to AbsoluteTo() to have same size.");

			if (dst.ModifyInHost(*this))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = abs(*pSrc);
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.m_DeviceData.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				dst.GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync();
				StartAsync(dst.Stream());
				Launch_CUDAKernel(Kernel_RealAbsolute, blocks, threads, /*dynamic shared memory=*/ 0, dst.Stream(),
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				dst.AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename FinalType> inline FinalType RealImage<PixelType, FinalType>::Absolute(HostFlags Flags) {
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? FinalType::NewHostImage(Width(), Height(), Stream(), Flags) : FinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			AbsoluteTo(ret);
			return ret;
		}

		template<typename PixelType, typename FinalType> inline FinalType& RealImage<PixelType, FinalType>::AbsoluteInPlace()
		{
			if (ModifyInHost())
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pImg = GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pImg++) *pImg = abs(*pImg);
				}
			}
			else
			{
				auto format = m_DeviceData.GetCudaFormat();
				ToDevice(true);

				dim3 blocks, threads;
				GetSmallOpKernelParameters(blocks, threads);

				StartAsync();
				Launch_CUDAKernel(Kernel_RealAbsoluteInPlace, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					GetDeviceDataPtr(), format
				);
				AfterKernelLaunch();
			}
			return (FinalType&)*this;
		}

		#pragma endregion

		#pragma region "Absolute Value and Angle, Complex"

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType> 
		inline RealImage<RealPixelType, RealFinalType>& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::AbsoluteTo(RealImage<RealPixelType, RealFinalType>& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to AbsoluteTo() to have same size.");

			if (dst.ModifyInHost(*this))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = thrust::abs(*pSrc);
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				dst.GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync();
				StartAsync(dst.Stream());
				Launch_CUDAKernel(Kernel_ComplexAbsolute, blocks, threads, /*dynamic shared memory=*/ 0, dst.Stream(),
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				dst.AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType> 
		inline RealFinalType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::Absolute(HostFlags Flags) {
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? RealFinalType::NewHostImage(Width(), Height(), Stream(), Flags) : RealFinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			AbsoluteTo(ret);
			return ret;
		}		

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealImage<RealPixelType, RealFinalType>& ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::AngleTo(RealImage<RealPixelType, RealFinalType>& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to AngleTo() to have same size.");

			if (dst.ModifyInHost(*this))
			{
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					auto pDst = dst.GetHostScanlinePtr(yy);

					for (int xx = 0; xx < Width(); xx++, pSrc++, pDst++) *pDst = thrust::arg(*pSrc);
				}
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dim3 blocks, threads;
				dst.GetSmallOpKernelParameters(blocks, threads);

				dst.StartAsync();
				StartAsync(dst.Stream());
				Launch_CUDAKernel(Kernel_ComplexArgument, blocks, threads, /*dynamic shared memory=*/ 0, dst.Stream(),
					dst.GetDeviceDataPtr(), formatDst,
					GetDeviceDataPtr(), formatSrc
				);
				dst.AfterKernelLaunch();
			}
			return dst;
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		inline RealFinalType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::Angle(HostFlags Flags) {
			if (Flags == HostFlags::Retain) Flags = m_HostData.GetFlags();
			auto ret = WouldModifyInHost() ? RealFinalType::NewHostImage(Width(), Height(), Stream(), Flags) : RealFinalType::NewDeviceImage(Width(), Height(), Stream(), Flags);
			AngleTo(ret);
			return ret;
		}

		#pragma endregion		

		#pragma region "Arithmetic Sum, Mean, and StdDev"

		template<typename PixelType, typename FinalType> 
		template<typename SumType>
		inline SumType RealImage<PixelType, FinalType>::Sum(Rectangle<int> ROI)
		{
			if (ROI.IsWhole()) ROI = Bounds();
			if (!ROI.IsContainedIn(Bounds()))
				throw ArgumentException("ROI must fully fit within the image dimensions.");

			switch (m_DataState)
			{
			case DataState::Host:
			{
				SumType accum = (SumType)0.0;
				for (int yy = ROI.Top(); yy < ROI.Bottom(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					pSrc += ROI.Left();
					for (int xx = 0; xx < ROI.Width; xx++, pSrc++) accum += *pSrc;
				}
				return accum;
			}
			case DataState::HostAndDevice:
			case DataState::Device:
			{
				NppiSize oSizeROI = ROI;
				int BufferSize = 0;
				cudaThrowable(NPPI::Behaviors<PixelType>::GetSumRealBufferSize(oSizeROI, &BufferSize, Stream()));
				while (m_Stream.GetScratch().size() < 2) m_Stream.GetScratch().push_back(cuda::memory::DeviceScratchBuffer());
				m_Stream.GetScratch()[0].Allocate(BufferSize);
				m_Stream.GetScratch()[1].Allocate(sizeof(SumType));				
				PixelType* pSrc = (PixelType*)(((byte*)GetDeviceDataPtr()) + ROI.Top() * m_DeviceData.m_Stride);
				pSrc += ROI.Left();
				// Example usage here:  https://docs.nvidia.com/cuda/npp/general_conventions_lb.html#general_scratch_buffer
				cudaThrowable(NPPI::Behaviors<PixelType>::SumReal(pSrc, m_DeviceData.m_Stride, oSizeROI, 
					m_Stream.GetScratch()[0].GetDevicePtr(), (SumType*)m_Stream.GetScratch()[1].GetDevicePtr(), m_Stream));
				SumType accum = (SumType)0.0;
				cudaThrowable(cudaMemcpy(&accum, m_Stream.GetScratch()[1].GetDevicePtr(), sizeof(SumType), cudaMemcpyDeviceToHost));
				return accum;
			}
			default: throw Exception("Invalid data state.");
			}
		}

		template<typename PixelType, typename FinalType>
		template<typename MeanType>
		inline MeanType RealImage<PixelType, FinalType>::Mean(Rectangle<int> ROI) { return Sum<MeanType>(ROI) / ((MeanType)ROI.Width * (MeanType)ROI.Height); }

		template<typename PixelType, typename FinalType>		
		inline void RealImage<PixelType, FinalType>::MeanAndStdDev(double& dMean, double& dStdDev)
		{
			switch (m_DataState)
			{
			case DataState::Host:
			{
				dMean = Mean<double>();
				double accum = 0.0;
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					for (int xx = 0; xx < Width(); xx++, pSrc++) accum += ((*pSrc - dMean) * (*pSrc - dMean));
					dStdDev = sqrt(accum / (double)(Width() * Height()));
				}
				return;
			}
			case DataState::HostAndDevice:
			case DataState::Device:
			{
				NppiSize oSizeROI = Bounds();
				int BufferSize = 0;
				cudaThrowable(NPPI::Behaviors<PixelType>::GetMeanAndStdDevRealBufferSize(oSizeROI, &BufferSize, Stream()));
				while (m_Stream.GetScratch().size() < 3) m_Stream.GetScratch().push_back(cuda::memory::DeviceScratchBuffer());
				m_Stream.GetScratch()[0].Allocate(BufferSize);
				m_Stream.GetScratch()[1].Allocate(sizeof(double));
				m_Stream.GetScratch()[2].Allocate(sizeof(double));
				// Example usage here:  https://docs.nvidia.com/cuda/npp/general_conventions_lb.html#general_scratch_buffer
				cudaThrowable(NPPI::Behaviors<PixelType>::MeanAndStdDevReal(GetDeviceDataPtr(), m_DeviceData.m_Stride, oSizeROI,
					m_Stream.GetScratch()[0].GetDevicePtr(), (double*)m_Stream.GetScratch()[1].GetDevicePtr(), (double*)m_Stream.GetScratch()[2].GetDevicePtr(), m_Stream));				
				cudaThrowable(cudaMemcpy(&dMean, m_Stream.GetScratch()[1].GetDevicePtr(), sizeof(double), cudaMemcpyDeviceToHost));
				cudaThrowable(cudaMemcpy(&dStdDev, m_Stream.GetScratch()[2].GetDevicePtr(), sizeof(double), cudaMemcpyDeviceToHost));
				return;
			}
			default: throw Exception("Invalid data state.");
			}
		}

		// ComplexImage
#if 0	// I'm getting a compiler error about type name is not allowed.  I must have changed something somewhere else and it's having a trickle effect.  Disabling for now.
		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		template<typename SumType>
		inline SumType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::Sum()
		{
			switch (m_DataState)
			{
			case DataState::Host:
			{
				SumType accum = (SumType)0.0;
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					for (int xx = 0; xx < Width(); xx++, pSrc++) accum += *pSrc;
				}
				return accum;
			}
			case DataState::HostAndDevice:
			case DataState::Device:
			{
				//const int ItemsPerThread = 1;
				const int ThreadsPerBlockX = 32, ThreadsPerBlockY = 16;
				const dim3 threads(ThreadsPerBlockX, ThreadsPerBlockY, 1);
				int NBlocksX = divup(Width(), threads.x), NBlocksY = divup(Height(), threads.y);				
				dim3 blocks(NBlocksX, NBlocksY, 1);
				int ScratchBufferSize = NBlocksX * NBlocksY * sizeof(SumType);				
				if (m_Stream.GetScratch().size() < 1) m_Stream.GetScratch().push_back(cuda::memory::DeviceScratchBuffer());
				m_Stream.GetScratch()[0].Allocate(ScratchBufferSize);
				
				// Since the kernel is using atomicAdd() to combine the final result (for reasons that I don't understand)...
				cudaThrowable(cudaMemset(m_Stream.GetScratch()[0].GetDevicePtr(), 0, ScratchBufferSize));

				auto format = m_DeviceData.GetCudaFormat();

				StartAsync();
				Launch_CUDAKernel(Kernel_ComplexSumBlockReduce<SumType::value_type COMMA PixelType COMMA ThreadsPerBlockX COMMA ThreadsPerBlockY COMMA 1>, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
					GetDeviceDataPtr(), format,
					(SumType*)m_Stream.GetScratch()[0].GetDevicePtr()
				);
				AfterKernelLaunch();

				Synchronize();
				SumType* pHostScratch = new SumType[NBlocksX * NBlocksY];
				SumType accum = (SumType)0.0;
				try
				{
					cudaThrowable(cudaMemcpy(pHostScratch, m_Stream.GetScratch()[0].GetDevicePtr(), ScratchBufferSize, cudaMemcpyDeviceToHost));
					//cudaThrowable(cudaStreamSynchronize(Stream()));			// Since this is not an Async memcpy, I think this is unnecessary.  See:  https://docs.nvidia.com/cuda/npp/general_conventions_lb.html#general_scratch_buffer
					for (int ii = 0; ii < NBlocksX * NBlocksY; ii++) accum += pHostScratch[ii];
				}
				catch (...)
				{
					delete[] pHostScratch;
					throw;
				}
				delete[] pHostScratch;
				return accum;
			}
			default: throw Exception("Invalid data state.");
			}
		}

		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType>
		template<typename MeanType>
		inline MeanType ComplexImage<PixelType, RealPixelType, FinalType, RealFinalType>::Mean() { return Sum<MeanType>() / ((MeanType)Width() * (MeanType)Height()); }
#endif
		#pragma endregion

		#pragma region "Min and Max"

		template<typename PixelType, typename FinalType>
		inline PixelType RealImage<PixelType, FinalType>::Min()
		{
			switch (m_DataState)
			{
			case DataState::Host:
			{
				//Asserts floating point compatibility at compile time
				static_assert(!std::numeric_limits<PixelType>::has_infinity || std::numeric_limits<PixelType>::is_iec559, "IEEE 754 required");
				PixelType best = std::numeric_limits<PixelType>::has_infinity ? std::numeric_limits<PixelType>::infinity() : std::numeric_limits<PixelType>::max();
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					for (int xx = 0; xx < Width(); xx++, pSrc++) if (*pSrc < best) best = *pSrc;
				}
				return best;
			}
			case DataState::HostAndDevice:
			case DataState::Device:
			{
				NppiSize oSizeROI = Bounds();
				int BufferSize = 0;
				cudaThrowable(NPPI::Behaviors<PixelType>::GetMinRealBufferSize(oSizeROI, &BufferSize, Stream()));
				while (m_Stream.GetScratch().size() < 2) m_Stream.GetScratch().push_back(cuda::memory::DeviceScratchBuffer());
				m_Stream.GetScratch()[0].Allocate(BufferSize);
				m_Stream.GetScratch()[1].Allocate(sizeof(PixelType));
				// Example usage here:  https://docs.nvidia.com/cuda/npp/general_conventions_lb.html#general_scratch_buffer
				cudaThrowable(NPPI::Behaviors<PixelType>::MinReal(GetDeviceDataPtr(), m_DeviceData.m_Stride, oSizeROI,
					m_Stream.GetScratch()[0].GetDevicePtr(), (PixelType*)m_Stream.GetScratch()[1].GetDevicePtr(), m_Stream));
				PixelType best = (PixelType)0.0;
				cudaThrowable(cudaMemcpy(&best, m_Stream.GetScratch()[1].GetDevicePtr(), sizeof(PixelType), cudaMemcpyDeviceToHost));
				return best;
			}
			default: throw Exception("Invalid data state.");
			}
		}

		template<typename PixelType, typename FinalType>
		inline PixelType RealImage<PixelType, FinalType>::Max()
		{
			switch (m_DataState)
			{
			case DataState::Host:
			{
				//Asserts floating point compatibility at compile time
				static_assert(!std::numeric_limits<PixelType>::has_infinity || std::numeric_limits<PixelType>::is_iec559, "IEEE 754 required");				
				PixelType best = std::numeric_limits<PixelType>::has_infinity ? -std::numeric_limits<PixelType>::infinity() : std::numeric_limits<PixelType>::lowest();
				for (int yy = 0; yy < Height(); yy++)
				{
					auto pSrc = GetHostScanlinePtr(yy);
					for (int xx = 0; xx < Width(); xx++, pSrc++) if (*pSrc > best) best = *pSrc;
				}
				return best;
			}
			case DataState::HostAndDevice:
			case DataState::Device:
			{
				NppiSize oSizeROI = Bounds();
				int BufferSize = 0;
				cudaThrowable(NPPI::Behaviors<PixelType>::GetMaxRealBufferSize(oSizeROI, &BufferSize, Stream()));
				while (m_Stream.GetScratch().size() < 2) m_Stream.GetScratch().push_back(cuda::memory::DeviceScratchBuffer());
				m_Stream.GetScratch()[0].Allocate(BufferSize);
				m_Stream.GetScratch()[1].Allocate(sizeof(PixelType));
				// Example usage here:  https://docs.nvidia.com/cuda/npp/general_conventions_lb.html#general_scratch_buffer
				cudaThrowable(NPPI::Behaviors<PixelType>::MaxReal(GetDeviceDataPtr(), m_DeviceData.m_Stride, oSizeROI,
					m_Stream.GetScratch()[0].GetDevicePtr(), (PixelType*)m_Stream.GetScratch()[1].GetDevicePtr(), m_Stream));
				PixelType best = (PixelType)0.0;
				cudaThrowable(cudaMemcpy(&best, m_Stream.GetScratch()[1].GetDevicePtr(), sizeof(PixelType), cudaMemcpyDeviceToHost));
				return best;
			}
			default: throw Exception("Invalid data state.");
			}
		}

		#pragma endregion

		#pragma region "Filters/Convolution"

		template<typename PixelType, typename FinalType>
		inline FinalType& RealImage<PixelType, FinalType>::FilterSobelHorizontalTo(FinalType& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (dst.ModifyInHost(*this))
			{
				throw NotImplementedException("Sobel filter is only currently implemented device-side.");
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);				

				dst.StartAsync();
				StartAsync(dst.Stream());
				cudaThrowable(NPPI::Behaviors<PixelType>::FilterSobelHoriz(
					GetDeviceDataPtr(), m_DeviceData.m_Stride,
					dst.GetDeviceDataPtr(), dst.m_DeviceData.m_Stride,
					dst.Bounds(), dst.Stream()));
			}
			return dst;
		}

		template<typename PixelType, typename FinalType>
		inline FinalType& RealImage<PixelType, FinalType>::FilterSobelVerticalTo(FinalType& dst)
		{
			if (dst.Width() != Width() || dst.Height() != Height()) throw ArgumentException("Expected source and destination images to have same size.");

			if (dst.ModifyInHost(*this))
			{
				throw NotImplementedException("Sobel filter is only currently implemented device-side.");
			}
			else
			{
				auto formatSrc = m_DeviceData.GetCudaFormat();
				auto formatDst = dst.GetCudaFormat();
				if (formatSrc.size() != formatDst.size()) throw FormatException("Image sizes must match for this operation.");
				ToDevice(false);
				dst.ToDevice(true);

				dst.StartAsync();
				StartAsync(dst.Stream());
				cudaThrowable(NPPI::Behaviors<PixelType>::FilterSobelVert(
					GetDeviceDataPtr(), m_DeviceData.m_Stride,
					dst.GetDeviceDataPtr(), dst.m_DeviceData.m_Stride,
					dst.Bounds(), dst.Stream()));
			}
			return dst;
		}

		#pragma endregion

	}// namespace images
}// namespace wb

#endif	// __WBImages_Math_h__

//	End of Images_Math.h


