/////////
//	Images.h
/////////

#ifndef __WBImages_h__
#define __WBImages_h__

#include "../wbCore.h"
#include "Implementation/Images_Base.h"

namespace wb
{
	namespace images
	{
		/** References **/

		using namespace wb::cuda;				
		template<typename PixelType> class ConvolutionKernel;				// Forward declaration

		#pragma region "Image Specializations"

		template<typename PixelType> class Image : public RealImage<PixelType, Image<PixelType> > 
		{ 
			typedef RealImage<PixelType, Image<PixelType>> base;
			friend BaseImage<PixelType, Image<PixelType>>;
			friend class base;
			
		protected:
			Image(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			Image() = default;
			Image(Image&&) = default;
			Image(Image&) = delete;
			Image& operator=(Image&&) = default;
			Image& operator=(Image&) = delete;
		};

		#pragma region "Byte, UInt16, UInt32 Images"

		template<> class Image<byte> : public RealImage<byte, Image<byte>>
		{
			typedef byte PixelType;
			typedef RealImage<PixelType, Image<PixelType>> base;
			friend BaseImage<PixelType, Image<PixelType>>;
			friend Image<float>;
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
			void Save(const string& sFilename);
			void Save(wb::io::Stream& Stream, FileFormat format);
		};

		template<> class Image<UInt16> : public RealImage<UInt16, Image<UInt16>>
		{
			typedef UInt16 PixelType;
			typedef RealImage<PixelType, Image<PixelType>> base;
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

			//static Image<UInt16> CopyFrom(const Image<float>& Src);
			//static Image<UInt16> CopyFrom(const Image<double>& Src);
			//static Image<UInt16> CopyFrom(const Image<int>& Src);

			Image<float>& ConvertTo(Image<float>& dst);
			template<typename PixelTypeDst> Image<PixelTypeDst> ConvertTo(HostFlags Flags = HostFlags::Retain);

			static Image Load(const std::string& filename, GPUStream Stream = GPUStream::None());
			void Save(const std::string& filename);
		};		

		#pragma endregion

		#pragma region "Floating-Point Images"

		template<>
		class Image<float> : public RealImage<float, Image<float>>
		{
			typedef float PixelType;
			typedef RealImage<PixelType, Image<PixelType>> base;
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

			typedef Image_Range<float> Range;	

			/// <summary>Image conversion.  Provide the value range from the source image that will be mapped to 0...255 in the destination image, or
			/// use the default of 0...1.</summary>
			Image<byte>& ConvertTo(Image<byte>& dst, Range SrcValueRange = Range::Unit);
			template<typename ToPixelType> Image<ToPixelType> ConvertTo(Range SrcValueRange = Range::Unit, HostFlags flags = HostFlags::Retain);
			Image<double>& ConvertTo(Image<double>& dst);

			static Image Load(const std::string& filename, GPUStream Stream = GPUStream::None());
			void Save(const std::string& filename);
			void Save(wb::io::Stream& Stream, FileFormat Format);
		};

		template<>
		class Image<double> : public RealImage<double, Image<double>>
		{
			typedef double PixelType;
			typedef RealImage<PixelType, Image<PixelType>> base;
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

			typedef Image_Range<double> Range;			

			/** Not sure I should have implicit conversions without a named conversion function...
			Image(const BaseImage<UInt16>& cvt) : BaseImage(cvt.Width(), cvt.Height())
			{
				for (int yy = 0; yy < cvt.Height(); yy++)
				{
					UInt16* pSrcScan = (UInt16*)cvt.GetScanlinePtr(yy);
					double* pDstScan = (double*)GetScanlinePtr(yy);
					for (int xx = 0; xx < cvt.Width(); xx++) pDstScan[xx] = (double)pSrcScan[xx];
				}
			}
			**/
			
			Image<float>& ConvertTo(Image<float>& dst);

			static Image Load(const std::string& filename, GPUStream Stream = GPUStream::None());
			void Save(const std::string& filename);
			void Save(wb::io::Stream& Stream, FileFormat Format);
		};		

		#pragma endregion		

	}// namespace images
}// namespace wb

//	Late Dependencies

#include "Implementation/Images_Math.h"
#include "Implementation/Images_Color.h"
#include "Implementation/Images_Conversions.h"
#include "Implementation/Images_LoadSave.h"

#endif	// __WBImages_h__

//	End of Images.h

