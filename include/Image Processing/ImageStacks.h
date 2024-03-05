/////////
//	ImageStacks.h
/////////

#ifndef __WBImageStacks_h__
#define __WBImageStacks_h__

#include "../wbFoundation.h"
#include "../wbCore.h"

#include "../System/GPU.h"			// Includes cuda.h, if supported.  Defines empty but convenient GPUStream if not supported.
#include "Images.h"

namespace wb
{
	namespace images
	{
		/** References **/
		using namespace wb::cuda;								

		#pragma region "Image Stacks"

		template<typename PixelType> class ImageStack : public std::vector<Image<PixelType>>
		{
			typedef std::vector<Image<PixelType>> base;			

		public:

			#pragma region "File Formats"

			#ifdef FreeImage_Support

			static ImageStack<PixelType> Load(const osstring& filename, GPUStream Stream = GPUStream::None())
			{
				auto ret = ImageStack<PixelType>();
				FreeImage_SetOutputMessage(FI::ErrorHandler);
				FREE_IMAGE_FORMAT fif = FI::GetFormat(wb::io::Path::GetExtension(wb::to_string(filename)));
				//FILE* pFile = _tfopen(filename, "rb");
				//auto pMB = FreeImage_OpenMultiBitmapFromHandle(fif, &io, (fi_handle)pFile, flags);
				auto pMB = FreeImage_OpenMultiBitmap(fif, wb::to_string(filename).c_str(), false, true, false);
				try
				{
					int count = FreeImage_GetPageCount(pMB);										
					for (int ii = 0; ii < count; ii++)
					{
						auto pFIB = FreeImage_LockPage(pMB, ii);
						try
						{
							if (pFIB == nullptr) throw IOException(string("Unable to load page [") + wb::to_string(ii) + "] from image '" + wb::to_string(filename) + "': unspecified error");
							if (FreeImage_GetBits(pFIB) == nullptr) throw IOException("Unable to load image '" + wb::to_string(filename) + "': image is empty and contains no pixel data");
							if (FreeImage_GetBPP(pFIB) != 8 * sizeof(PixelType))
								throw FormatException("Image '" + wb::to_string(filename) + "': is a " + std::to_string(FreeImage_GetBPP(pFIB)) + " bpp image but a " + std::to_string(8 * sizeof(PixelType)) + " bpp image was expected.");
							auto img = Image<PixelType>::ExistingHostImageWrapper(FreeImage_GetWidth(pFIB), FreeImage_GetHeight(pFIB), FreeImage_GetPitch(pFIB), (PixelType*)FreeImage_GetBits(pFIB), true, Stream, HostFlags::None);							
							auto Flipped = Image<PixelType>::NewHostImage(img.Width(), img.Height(), Stream, HostFlags::None);
							img.FlipVerticallyTo(Flipped);
							ret.push_back(std::move(Flipped));
						}
						catch(...)
						{
							FreeImage_UnlockPage(pMB, pFIB, FALSE);
							throw;
						}
						FreeImage_UnlockPage(pMB, pFIB, FALSE);
					}					
				}
				catch(...)
				{
					FreeImage_CloseMultiBitmap(pMB, 0);
					throw;
				}
				FreeImage_CloseMultiBitmap(pMB, 0);
				return ret;
			}

			#endif		

			#pragma endregion		

			#pragma region "Construction and Host-Device Management"

		public:

			ImageStack()
			{
			}			

			ImageStack(ImageStack&) = delete;
			ImageStack& operator=(ImageStack&) = delete;
			
			ImageStack(ImageStack<PixelType>&& mv) noexcept
				: base(std::move(mv))
			{				
			}

			ImageStack& operator=(ImageStack<PixelType>&& mv)
			{
				base::operator=(std::move(mv));				
				return *this;
			}

			static ImageStack NewHostStack(int Width, int Height, int NFrames, GPUStream Stream = GPUStream::None(), HostFlags Flags = HostFlags::None)
			{				
				ImageStack<PixelType> ret(NFrames);
				for (int ii = 0; ii < NFrames; ii++)
					ret.push_back(Image<PixelType>::NewHostImage(Width, Height, Stream, Flags));				
				return ret;
			}

			#ifdef CUDA_Support
			static ImageStack NewDeviceImage(int Width, int Height, int NFrames, GPUStream Stream, HostFlags Flags = HostFlags::Pinned)
			{
				ImageStack<PixelType> ret(NFrames);
				for (int ii = 0; ii < NFrames; ii++)
					ret.push_back(Image<PixelType>::NewDeviceImage(Width, Height, Stream, Flags));
				return ret;
			}
			#endif		

			#pragma endregion

			#pragma region "Attributes"

			int Width() const { return at(0).Width(); }
			int Height() const { return at(0).Height(); }
			int NFrames() const { return (int)base::size(); }

			#pragma endregion
		};		

		#pragma region "Troubleshooting & Tools"		

		template<typename PixelType> inline std::string to_string(const ImageStack<PixelType>& src)
		{
			std::stringstream out;
			out.precision(3);
			out << "image_stack(width=" << src.Width() << ", height=" << src.Height() << ", n_frames=" << src.NFrames() << "): \n";

			/*
			#define write_line(yy)			\
				{							\
					out << "\t";			\
					int x_show = (src.Width() <= 10 ? src.Width() : 3);						\
					for (int xx = 0; xx < x_show; xx++) {		\
						out << display_string(src(xx, yy));									\
						if (src.Width() > xx + 1) out << ", ";								\
					}						\
					if (src.Width() > 10) {													\
						size_t curr_len = out.str().rfind('\n');							\
						if (curr_len == string::npos) curr_len = 0;							\
						curr_len = out.str().size() - curr_len;								\
						for (size_t ii = curr_len; ii < 40; ii++) out << " ";				\
						out << "...,        ";												\
						for (int xx = src.Width() - 3; xx < src.Width(); xx++) {			\
							out << display_string(src(xx, yy));								\
							if (src.Width() > xx + 1) out << ", ";							\
						}					\
					}						\
					out << "\n";			\
				}

			int y_show = (src.Height() <= 10 ? src.Height() : 3);
			for (int yy = 0; yy < y_show; yy++) write_line(yy);			
			if (src.Height() > 10) {
				out << "\t  ...\n";
				for (int yy = src.Height() - 3; yy < src.Height(); yy++) write_line(yy);
			}

			#undef write_line
			*/
			return out.str();
		}

		template<typename PixelType> inline std::ostream& operator<<(std::ostream& os, const ImageStack<PixelType>& img)
		{
			os << to_string(img);
			return os;
		}

		#pragma endregion

	}// namespace images
}// namespace wb

/////////
//	Late Dependencies
//

#endif	// __WBImageStacks_h__

//	End of ImageStacks.h

