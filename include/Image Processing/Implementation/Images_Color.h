/////////
//	Images_Color.h
/////////

#ifndef __WBImages_Color_h__
#define __WBImages_Color_h__

#ifndef __WBImages_h__
#error Include this header only via Images.h.
#endif

#include "../Images.h"

namespace wb
{
	namespace images
	{
		/** References **/

		#pragma region "Generalized Color Image Template"

		/// <summary>
		/// ColorImage (multiple-channel image) is an intermediate class.  Descending from BaseImage,
		/// it provides functionality that is specific to color, multiple-channel images only.  Examples
		/// of classes that descend from ColorImage are Image&lt;RGBPixel&gt; and 
		/// Image&lt;RGBAPixel&gt;.
		/// </summary>		
		template<typename PixelType, typename FinalType> class ColorImage : public BaseImage<PixelType, FinalType>
		{
			typedef BaseImage<PixelType, FinalType> base;

		protected:
			ColorImage(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			ColorImage() = default;
			ColorImage(ColorImage&&) = default;
			ColorImage(ColorImage&) = delete;
			ColorImage& operator=(ColorImage&&) = default;
			ColorImage& operator=(ColorImage&) = delete;
		};

		/// <summary>
		/// RGBColorImage is an intermediate class.  Descending from ColorImage, it provides functionality that is 
		/// specific to color, multiple-channel images in RGB or RGBA formats.  Examples of classes that descend 
		/// from ColorImage are Image&lt;RGBPixel&gt; and Image&lt;RGBAPixel&gt;.
		/// </summary>		
		template<typename PixelType, typename FinalType> class RGBColorImage : public ColorImage<PixelType, FinalType>
		{
			typedef ColorImage<PixelType, FinalType> base;

		protected:
			RGBColorImage(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			RGBColorImage() = default;
			RGBColorImage(RGBColorImage&&) = default;
			RGBColorImage(RGBColorImage&) = delete;
			RGBColorImage& operator=(RGBColorImage&&) = default;
			RGBColorImage& operator=(RGBColorImage&) = delete;

			template<typename NewPixelType> Image<NewPixelType>& ConvertToGrayscale(Image<NewPixelType>& Grayscale);
			template<typename NewPixelType> Image<NewPixelType> ConvertToGrayscale(HostFlags Flags = HostFlags::Retain);
		};

		#pragma endregion

		#pragma region "Color Image Specializations"

		template<>
		class Image<RGBPixel> : public RGBColorImage<RGBPixel, Image<RGBPixel>>
		{
			typedef RGBPixel PixelType;
			typedef RGBColorImage<PixelType, Image<PixelType>> base;
			friend class BaseImage<PixelType, Image<PixelType>>;			
			friend class base;
			friend class AviWriter;
			friend class Image<RGBAPixel>;			

		protected:
			Image(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			Image() = default;
			Image(Image&&) = default;
			Image(Image&) = delete;
			Image& operator=(Image&&) = default;
			Image& operator=(Image&) = delete;
		
			Image<RGBAPixel>& ConvertTo(Image<RGBAPixel>& dst, byte Alpha = 255);
			template<typename ToPixelType> Image<ToPixelType> ConvertTo(byte Alpha = 255, HostFlags HostFlags = HostFlags::Retain);

			static Image<RGBPixel> Load(const osstring& filename, GPUStream Stream = GPUStream::None());
			void Save(const osstring& filename);
			void Save(wb::io::Stream& Stream, FileFormat Format);			
		};

		template<>
		class Image<RGBAPixel> : public RGBColorImage<RGBAPixel, Image<RGBAPixel>>
		{
			typedef RGBAPixel PixelType;
			typedef RGBColorImage<PixelType, Image<PixelType>> base;
			friend class BaseImage<PixelType, Image<PixelType>>;
			friend class base;
			friend class Image<RGBPixel>;
			friend class AviWriter;

		protected:
			Image(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			Image() = default;
			Image(Image&&) = default;
			Image(Image&) = delete;
			Image& operator=(Image&&) = default;
			Image& operator=(Image&) = delete;

			//static Image<RGBAPixel> CopyFrom(const Image<byte>&, byte Alpha = 255);
			//static Image<RGBAPixel> CopyFrom(const Image<float>&, Image<float>::Range ValueRange = Image<float>::UnitRange, byte Alpha = 255);
			//static Image<RGBAPixel> CopyFrom(const Image<double>&, Image<double>::Range ValueRange = Image<double>::UnitRange, byte Alpha = 255);
			static Image<RGBAPixel> CopyFrom(Image<RGBPixel>&, byte Alpha = 255);
			
			static Image<RGBAPixel> Load(const osstring& filename, GPUStream Stream = GPUStream::None());
			static Image<RGBAPixel> Load(wb::io::Stream& Stream, FileFormat Format, GPUStream gpuStream = GPUStream::None());
			void Save(const osstring& filename);
			void Save(wb::io::Stream& Stream, FileFormat Format);						

			#if 0
			enum class Channel
			{
				R, G, B, A
			};

			void ExtractChannelTo(Channel iChannel, Image<byte>& Channel) const
			{
				if (!Channel.IsAllocated()) Channel.Allocate(Width(), Height());
				if (Channel.Width() != Width() || Channel.Height() != Height())
					throw ArgumentException("Destination channel image must be unallocated or match source image size.");

				for (int sy = 0; sy < m_Height; sy++)
				{
					const RGBAPixel* pSrcScanline = (const RGBAPixel*)GetScanlinePtr(sy);
					byte* pDstScanline = (byte*)Channel.GetScanlinePtr(sy);

					for (int xx = 0; xx < m_Width; xx++, pSrcScanline++, pDstScanline++)
					{
						switch (iChannel)
						{
						case Channel::R: *pDstScanline = pSrcScanline->R; break;
						case Channel::G: *pDstScanline = pSrcScanline->G; break;
						case Channel::B: *pDstScanline = pSrcScanline->B; break;
						case Channel::A: *pDstScanline = pSrcScanline->A; break;
						}
					}
				}
			}

			void ReplaceChannelWith(Channel iChannel, const Image<byte>& Channel) const
			{
				if (Channel.Width() != Width() || Channel.Height() != Height())
					throw ArgumentException("Destination channel image must match target image size.");

				for (int sy = 0; sy < m_Height; sy++)
				{
					const byte* pSrcScanline = (const byte*)Channel.GetScanlinePtr(sy);
					RGBAPixel* pDstScanline = (RGBAPixel*)GetScanlinePtr(sy);

					for (int xx = 0; xx < m_Width; xx++, pSrcScanline++, pDstScanline++)
					{
						switch (iChannel)
						{
						case Channel::R: pDstScanline->R = *pSrcScanline; break;
						case Channel::G: pDstScanline->G = *pSrcScanline; break;
						case Channel::B: pDstScanline->B = *pSrcScanline; break;
						case Channel::A: pDstScanline->A = *pSrcScanline; break;
						}
					}
				}
			}
			#endif
		};

		#pragma endregion		

	}// namespace images
}// namespace wb

#endif	// __WBImages_Color_h__

//	End of Images_Color.h


