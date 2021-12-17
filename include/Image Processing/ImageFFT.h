/////////
//	ImageFFT.h
/////////

#ifndef __WBImageFFT_h__
#define __WBImageFFT_h__

#include "../wbCore.h"
#include "Images.h"
#include "../System/GPU.h"

#ifdef cuFFT_Support

namespace wb
{
	namespace images
	{
		using namespace wb::cuda;

		enum class FFTType
		{
			ComplexToComplex = CUFFT_C2C
		};

		template<typename PrecisionType> class FFTPlanBase
		{
		protected:

			cufftHandle		m_Plan;
			cufftType_t		m_Type;						// Set to 0x00 if m_Plan is invalid/destroyed.
			int				m_Width, m_Height;
			GPUStream		m_Stream;

			void Create(int Width, int Height, cufftType_t Type, memory::DeviceImageData<thrust::complex<PrecisionType>>* pWorkspace = nullptr)
			{
				m_Type = (cufftType_t)0x00;
				m_Width = Width;
				m_Height = Height;				
				if (Type != CUFFT_C2C && Type != CUFFT_Z2Z) throw NotImplementedException("Only C2C FFT is currently implemented.");
				
				cudaThrowable(cufftCreate(&m_Plan));
				m_Type = Type;
				
				cudaThrowable(cufftSetAutoAllocation(m_Plan, pWorkspace == nullptr ? 1 : 0));		// Set False if we will manage the work areas ourselves.
				
				cudaThrowable(cufftSetStream(m_Plan, m_Stream));

				size_t work_size;
				cudaThrowable(cufftMakePlan2d(m_Plan, Height, Width, m_Type, &work_size));

				if (pWorkspace != nullptr)
				{
					if (work_size > pWorkspace->m_Width * pWorkspace->m_Height * sizeof(thrust::complex<PrecisionType>)) throw ArgumentException("Provided workspace for FFTPlan must be large enough to meet requirements.");

					// Requesting minimal space fails as of CUDA 11.0 for typical sizes
					//cudaThrowable(cufftXtSetWorkAreaPolicy(plan, CUFFT_WORKAREA_MINIMAL, &work_size));
					cudaThrowable(cufftSetWorkArea(m_Plan, pWorkspace->m_pData));
				}
			}

		public:
			FFTPlanBase() : m_Stream(GPUStream::None())
			{
				m_Type = (cufftType_t)0x00;
			}

			FFTPlanBase(GPUStream Stream) : m_Stream(Stream)
			{
				m_Type = (cufftType_t)0x00;
			}

			FFTPlanBase(FFTPlanBase&& mv) : m_Stream(std::move(mv.m_Stream))
			{
				m_Plan = mv.m_Plan;
				mv.m_Plan = (cufftHandle)0;
				m_Type = mv.m_Type;
				mv.m_Type = (cufftType_t)0x00;
				m_Width = mv.m_Width;
				m_Height = mv.m_Height;
			}

			FFTPlanBase& operator=(FFTPlanBase&& mv) noexcept
			{
				if (m_Type != 0x00)
				{
					cudaThrowable(cufftDestroy(m_Plan));
					m_Type = (cufftType_t)0x00;
				}

				m_Plan = mv.m_Plan;
				mv.m_Plan = (cufftHandle)0;
				m_Type = mv.m_Type;
				mv.m_Type = (cufftType_t)0x00;
				m_Width = mv.m_Width;
				m_Height = mv.m_Height;
				m_Stream = std::move(mv.m_Stream);
				return *this;
			}

			FFTPlanBase(const FFTPlanBase&) = delete;
			FFTPlanBase& operator=(const FFTPlanBase&) = delete;

			~FFTPlanBase()
			{
				if (m_Type != 0x00)
				{
					cudaThrowable(cufftDestroy(m_Plan));
					m_Type = (cufftType_t)0x00;
				}
			}

			virtual void Forward(Image<thrust::complex<PrecisionType>>& dst, Image<thrust::complex<PrecisionType>>& src) { }
			virtual void Inverse(Image<thrust::complex<PrecisionType>>& dst, Image<thrust::complex<PrecisionType>>& src) { }

			Image<thrust::complex<PrecisionType>> Forward(Image<thrust::complex<PrecisionType>>& src, HostFlags Flags = HostFlags::Retain)
			{
				if (Flags == HostFlags::Retain) Flags = src.GetHostFlags();
				auto ret = Image<thrust::complex<PrecisionType>>::NewDeviceImage(src.Width(), src.Height(), m_Stream, Flags);
				Forward(ret, src);
				return ret;
			}

			void ForwardInPlace(Image<thrust::complex<PrecisionType>>& img) {
				Forward(img, img);
			}						

			Image<thrust::complex<PrecisionType>> Inverse(Image<thrust::complex<PrecisionType>>& src, HostFlags Flags = HostFlags::Retain)
			{
				if (Flags == HostFlags::Retain) Flags = src.GetHostFlags();
				auto ret = Image<thrust::complex<PrecisionType>>::NewDeviceImage(src.Width(), src.Height(), m_Stream, Flags);
				Inverse(ret, src);
				return ret;
			}

			void InverseInPlace(Image<thrust::complex<PrecisionType>>& img) {
				Inverse(img, img);
			}
		};		

		template<typename PrecisionType> class FFTPlan : FFTPlanBase<PrecisionType>
		{
			typedef FFTPlanBase<PrecisionType> base;
		public:
		};

		template<> class FFTPlan<float> : public FFTPlanBase<float>
		{
			typedef FFTPlanBase<float> base;

		public:
			FFTPlan() : base()
			{
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream) : base(Stream)
			{
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_C2C, nullptr); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream, memory::DeviceImageData<thrust::complex<float>>& workspace) : base(Stream)
			{
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_C2C, &workspace); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream, Image<thrust::complex<float>>& workspace) : base(Stream)
			{
				workspace.ToDevice(true);
				workspace.SetStream(Stream);
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_C2C, &(workspace.m_DeviceData)); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}

			FFTPlan& operator=(FFTPlan&& mv) noexcept = default;
			FFTPlan(const FFTPlan&) = delete;
			FFTPlan& operator=(const FFTPlan&) = delete;

			void Forward(Image<thrust::complex<float>>& dst, Image<thrust::complex<float>>& src) override
			{
				if (m_Type != CUFFT_C2C) throw ArgumentException("Source and destination images of type thrust::complex<float> must use C2C type FFT.");
				src.ToDevice(false);
				dst.ToDevice(true);
				src.StartAsync(m_Stream);
				dst.StartAsync(m_Stream);
				cudaThrowable(cufftExecC2C(m_Plan, (cuFloatComplex*)src.GetDeviceDataPtr(), (cuFloatComplex*)dst.GetDeviceDataPtr(), CUFFT_FORWARD));
			}

			void Inverse(Image<thrust::complex<float>>& dst, Image<thrust::complex<float>>& src) override
			{
				if (m_Type != CUFFT_C2C) throw ArgumentException("Source and destination images of type thrust::complex<float> must use C2C type FFT.");
				src.ToDevice(false);
				dst.ToDevice(true);
				src.StartAsync(m_Stream);
				dst.StartAsync(m_Stream);
				cudaThrowable(cufftExecC2C(m_Plan, (cuFloatComplex*)src.GetDeviceDataPtr(), (cuFloatComplex*)dst.GetDeviceDataPtr(), CUFFT_INVERSE));
			}
		};

		template<> class FFTPlan<double> : public FFTPlanBase<double>
		{
			typedef FFTPlanBase<double> base;

		public:
			FFTPlan() : base()
			{
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream) : base(Stream)
			{
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_Z2Z, nullptr); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream, memory::DeviceImageData<thrust::complex<double>>& workspace) : base(Stream)
			{
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_Z2Z, &workspace); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}

			FFTPlan(int Width, int Height, FFTType Type, GPUStream Stream, Image<thrust::complex<double>>& workspace) : base(Stream)
			{
				workspace.ToDevice(true);
				workspace.SetStream(Stream);
				switch (Type)
				{
				case FFTType::ComplexToComplex: Create(Width, Height, CUFFT_Z2Z, &(workspace.m_DeviceData)); break;
				default: throw NotSupportedException("FFTType requested is not supported.");
				}
			}
			
			FFTPlan& operator=(FFTPlan&& mv) noexcept = default;
			FFTPlan(const FFTPlan&) = delete;
			FFTPlan& operator=(const FFTPlan&) = delete;

			void Forward(Image<thrust::complex<double>>& dst, Image<thrust::complex<double>>& src)
			{
				if (m_Type != CUFFT_Z2Z) throw ArgumentException("Source and destination images of type thrust::complex<double> must use Z2Z type FFT.");
				src.ToDevice(false);
				dst.ToDevice(true);				
				src.StartAsync(m_Stream);
				dst.StartAsync(m_Stream);
				cudaThrowable(cufftExecZ2Z(m_Plan, (cuDoubleComplex*)src.GetDeviceDataPtr(), (cuDoubleComplex*)dst.GetDeviceDataPtr(), CUFFT_FORWARD));
			}

			void Inverse(Image<thrust::complex<double>>& dst, Image<thrust::complex<double>>& src)
			{
				if (m_Type != CUFFT_Z2Z) throw ArgumentException("Source and destination images of type thrust::complex<double> must use Z2Z type FFT.");
				src.ToDevice(false);
				dst.ToDevice(true);
				src.StartAsync(m_Stream);
				dst.StartAsync(m_Stream);
				cudaThrowable(cufftExecZ2Z(m_Plan, (cuDoubleComplex*)src.GetDeviceDataPtr(), (cuDoubleComplex*)dst.GetDeviceDataPtr(), CUFFT_INVERSE));
			}
		};

	}// namespace images
}// namespace wb

#endif	// cuFFT_Support

#endif	// __WBImageFFT_h__

//	End of ImageFFT.h


