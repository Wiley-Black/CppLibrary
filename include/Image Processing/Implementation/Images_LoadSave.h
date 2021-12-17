/////////
//	Images_LoadSave.h
/////////

#ifndef __WBImages_LoadSave_h__
#define __WBImages_LoadSave_h__

#ifndef __WBImages_h__
#error Include this header only via Images.h.
#endif

#include <complex>
#include "../Images.h"

// FreeImage.h and .lib already referenced by Images_Memory.h.

namespace wb
{
	namespace images
	{
		#pragma region "General Image Load/Save Routines and Helpers"

		#ifdef FreeImage_Support
		namespace FI
		{
			template<typename PixelType> inline Image<PixelType> LoadHelper(const string& filename, GPUStream Stream)
			{
				FreeImage_SetOutputMessage(ErrorHandler);
				FREE_IMAGE_FORMAT fif = GetFormat(wb::io::Path::GetExtension(filename));
				FIBITMAP* pFIB;
				try
				{
					#ifdef _WIN32
					pFIB = FreeImage_LoadU(fif, to_osstring(filename).c_str(), 0);
					#else
					pFIB = FreeImage_Load(fif, filename.c_str(), 0);
					#endif
				}
				catch (std::exception& ex)
				{
					throw IOException("While loading image '" + filename + "': " + string(ex.what()));
				}
				if (pFIB == nullptr) throw IOException("Unable to load image '" + filename + "': unspecified error");
				if (FreeImage_GetBits(pFIB) == nullptr) throw IOException("Unable to load image '" + filename + "': image is empty and contains no pixel data");
				if (FreeImage_GetBPP(pFIB) != 8 * sizeof(PixelType)) 
					throw FormatException("Image '" + filename + "': is a " + std::to_string(FreeImage_GetBPP(pFIB)) + " bpp image but a " + std::to_string(8 * sizeof(PixelType)) + " bpp image was expected.");
				auto img = Image<PixelType>::ExistingHostImageWrapper(FreeImage_GetWidth(pFIB), FreeImage_GetHeight(pFIB), FreeImage_GetPitch(pFIB), (PixelType*)FreeImage_GetBits(pFIB), true, Stream, HostFlags::None);
				// The ExistingHostImageWrapper object will not take responsibility for freeing the bitmap memory (FreeImage_GetBits()) and we don't want it to.  However, the
				// HostImageData object within it can be made to take care of the FIBITMAP* and can be responsible for freeing it.  In a slightly more optimized world, we could
				// make this a descendant of HostImageData that carries around the file data to save the 8 bytes being carried around by all images, but it's only 8 bytes.
				img.m_HostData.m_pFileData = pFIB;
				auto Flipped = Image<PixelType>::NewHostImage(img.Width(), img.Height(), Stream, HostFlags::None);
				img.FlipVerticallyTo(Flipped);
				return Flipped;
			}

			template<typename PixelType, FREE_IMAGE_COLOR_TYPE FICT, UInt32 RedMask, UInt32 GreenMask, UInt32 BlueMask> inline Image<PixelType> ColorLoadHelper(const string& filename, GPUStream Stream)
			{
				auto ret = LoadHelper<PixelType>(filename, Stream);
				auto imgColorType = FreeImage_GetColorType(ret.m_HostData.m_pFileData);
				if (imgColorType != FICT) throw FormatException("Image '" + filename + "': does not match expected color format.");

				auto FileRMask = FreeImage_GetRedMask(ret.m_HostData.m_pFileData);
				auto FileGMask = FreeImage_GetGreenMask(ret.m_HostData.m_pFileData);
				auto FileBMask = FreeImage_GetBlueMask(ret.m_HostData.m_pFileData);

				/* We can perform an in-place BGR->RGB conversion if called for **/
				if (FileRMask == BlueMask && FileGMask == GreenMask && FileBMask == RedMask && sizeof(PixelType) == 3)
				{
					for (int yy = 0; yy < ret.Height(); yy++)
					{
						PixelType* pImg = ret.GetHostScanlinePtr(yy);
						for (int xx = 0; xx < ret.Width(); xx++, pImg++)
						{
							PixelType tmp = *pImg;
							pImg->R = tmp.B;
							pImg->B = tmp.R;
						}
					}

					auto tmp = FileBMask;
					FileBMask = FileRMask;
					FileRMask = tmp;
				}

				if (FileRMask != RedMask || FileGMask != GreenMask || FileBMask != BlueMask) throw FormatException("Image '" + filename + "': does not match expected color format (masks).");
				return ret;
			}

			inline FREE_IMAGE_FORMAT GetFormat(const string& file_extension)
			{
				string ext = to_lower(file_extension);
				if (ext.length() > 1 && ext[0] != '.') throw FormatException("GetFormat() requires that the first character of the file extension be a dot.");

				if (IsEqual(ext, ".bmp")) return FIF_BMP;		// Windows or OS / 2 Bitmap File(*.BMP)
				if (IsEqual(ext, ".dds")) return FIF_DDS;		// DirectDraw Surface(*.DDS)
				if (IsEqual(ext, ".exr")) return FIF_EXR;		// ILM OpenEXR(*.EXR)
				if (IsEqual(ext, ".g3")) return FIF_FAXG3;		// Raw Fax format CCITT G3(*.G3)
				if (IsEqual(ext, ".gif")) return FIF_GIF;		// Graphics Interchange Format(*.GIF)
				if (IsEqual(ext, ".hdr")) return FIF_HDR;		// High Dynamic Range(*.HDR)
				if (IsEqual(ext, ".ico")) return FIF_ICO;		// Windows Icon(*.ICO)
				if (IsEqual(ext, ".j2k") || IsEqual(ext, ".j2c")) return FIF_J2K;		// JPEG - 2000 codestream(*.J2K, *.J2C)
				if (IsEqual(ext, ".jng")) return FIF_JNG;		// JPEG Network Graphics(*.JNG)
				if (IsEqual(ext, ".jp2")) return FIF_JP2;		// JPEG - 2000 File Format(*.JP2)
				if (IsEqual(ext, ".jpg") || IsEqual(ext, ".jif") || IsEqual(ext, ".jpeg") || IsEqual(ext, ".jpe")) return FIF_JPEG;		// Independent JPEG Group(*.JPG, *.JIF, *.JPEG, *.JPE)
				if (IsEqual(ext, ".jxr") || IsEqual(ext, ".wdp") || IsEqual(ext, ".hdp")) return FIF_JXR;		// JPEG XR image format(*.JXR, *.WDP, *.HDP)
				if (IsEqual(ext, ".png")) return FIF_PNG;		// Portable Network Graphics(*.PNG)
				if (IsEqual(ext, ".psd")) return FIF_PSD;		// Adobe Photoshop(*.PSD)
				// .raw can mean RAW camera images, but those contain a header file.  .raw can also mean other formats, usually ad-hoc.
				// To avoid confusion, I'm excluding it from being recognized solely by file extension.
				if (IsEqual(ext, ".tif") || IsEqual(ext, ".tiff")) return FIF_TIFF;		// Tagged Image File Format(*.TIF, *.TIFF)
				if (IsEqual(ext, ".webp")) return FIF_WEBP;		// Google WebP image format(*.WEBP)
				return FIF_UNKNOWN;
			}			
		}
		#endif

		template<typename PixelType, typename FinalType> inline /*static*/ FinalType BaseImage<PixelType, FinalType>::LoadGeneric(const string& filename, GPUStream Stream)
		{
			#ifdef FreeImage_Support
			return FI::LoadHelper<PixelType>(filename, Stream);
			#endif

			throw FormatException("Unsupported file format.");
		}

		template<typename PixelType, typename FinalType> inline void BaseImage<PixelType, FinalType>::SaveGeneric(const string& filename, bool ApplyCompression)
		{
			#ifdef FreeImage_Support
			FreeImage_SetOutputMessage(FI::ErrorHandler);
			FREE_IMAGE_FORMAT fif = FI::GetFormat(wb::to_string(wb::io::Path::GetExtension(filename)));

			// FreeImage needs the image to be flipped vertically, and needs the data on the host side.
			// Note: the HostFlags::None is required to cause the host image data to be allocated using
			// FreeImage.
			auto Flipped = FinalType::NewHostImage(Width(), Height(), m_Stream, HostFlags::None);
			FlipVerticallyTo(Flipped);
			Flipped.Synchronize();
			if (Flipped.m_HostData.m_pFileData == nullptr) throw Exception("Expected FreeImage to be in-use for this image buffer because HostFlags::None specified.");
				
			int flags = 0;
			if (fif == FIF_TIFF) flags = ApplyCompression ? TIFF_DEFAULT : TIFF_NONE;

			#ifdef _WIN32
			bool Success = FreeImage_SaveU(fif, Flipped.m_HostData.m_pFileData, to_osstring(filename).c_str(), flags);
			#else
			bool Success = FreeImage_Save(fif, Flipped.m_HostData.m_pFileData, filename.c_str(), flags);
			#endif
			if (!Success) throw IOException("Unable to save image to file '" + filename + "'.");
			#else
			throw FormatException("Unsupported file format.");
			#endif
		}

		#pragma endregion

		#pragma region "Image<byte> Load/Save"

		inline /*static*/ Image<byte> Image<byte>::Load(const string& filename, GPUStream Stream)
		{
			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<byte>::Save(const string& sFilename)
		{
			#ifdef FreeImage_Support
			SaveGeneric(sFilename);
			#else
			FileFormat format = ToFileFormat(sFilename);
			if (format == FileFormat::Unspecified) throw ArgumentException("Filename provided in image save operation did not provide a file extension.");
			if (format == FileFormat::Unrecognized) throw NotSupportedException(string("File extension '") + wb::to_string(wb::io::Path::GetExtension(sFilename)) + "' not recognized in image save operation.");
			wb::io::FileStream fs(sFilename, wb::io::FileMode::Create);
			Save(fs, format);
			#endif
		}

		inline void Image<byte>::Save(wb::io::Stream& Stream, FileFormat format)
		{
			if (format != FileFormat::BMP) throw NotSupportedException("File format " + to_string(format) + " is not supported for this image type with a stream.");

			/** Good resources:
			http://tipsandtricks.runicsoft.com/Cpp/BitmapTutorial.html
			http://stackoverflow.com/questions/3142349/drawing-on-8bpp-grayscale-bitmap-unmanaged-c
			**/

			ToHost(false);
			Synchronize();

			Int32 ImageSize = m_HostData.m_Stride * m_HostData.m_Height;

			BITMAPFILEHEADER bmfh;
			BITMAPINFOHEADER info;
			memset(&bmfh, 0, sizeof(BITMAPFILEHEADER));
			memset(&info, 0, sizeof(BITMAPINFOHEADER));

			bmfh.bfType = 0x4d42;       // 0x4d42 = 'BM'			
			bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * 256;
			bmfh.bfSize = bmfh.bfOffBits + ImageSize;

			info.biSize = sizeof(BITMAPINFOHEADER);
			info.biWidth = Width();
			info.biHeight = Height();
			info.biPlanes = 1;
			info.biBitCount = 8;
			info.biCompression = BI_RGB;
			info.biSizeImage = 0;
			info.biXPelsPerMeter = 0x0ec4;
			info.biYPelsPerMeter = 0x0ec4;
			info.biClrUsed = 256;
			info.biClrImportant = 0;

			// Map greyscale to RGB triplets...
			RGBQUAD bmiColors[256];
			for (int ii = 0; ii < 256; ii++)
			{
				bmiColors[ii].rgbRed = ii;
				bmiColors[ii].rgbGreen = ii;
				bmiColors[ii].rgbBlue = ii;
				bmiColors[ii].rgbReserved = 0;
			}

			Stream.Write(&bmfh, sizeof(BITMAPFILEHEADER));
			Stream.Write(&info, sizeof(BITMAPINFOHEADER));
			Stream.Write(bmiColors, sizeof(RGBQUAD) * 256);
			for (int yy = Height() - 1; yy >= 0; yy--) Stream.Write(GetHostScanlinePtr(yy), m_HostData.m_Stride);
		}

		#pragma endregion

		#pragma region "Image<UInt16> Load/Save"

		inline /*static*/ Image<UInt16> Image<UInt16>::Load(const std::string& filename, GPUStream Stream)
		{
			if (!io::File::Exists(filename.c_str())) throw FileNotFoundException("Image file not found.");
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				TIFFSetWarningHandler(nullptr);					// Disable warnings.

				TIFF* input_image = TIFFOpen(filename.c_str(), "r");
				uint32 Width, Height;
				uint32 BitsPerSample, SamplesPerPixel;
				TIFFGetField(input_image, TIFFTAG_IMAGEWIDTH, &Width);
				TIFFGetField(input_image, TIFFTAG_IMAGELENGTH, &Height);
				TIFFGetField(input_image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
				TIFFGetField(input_image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
				Image ret((int)Width, (int)Height);
				for (uint32 yy = 0; yy < Height; yy++) TIFFReadScanline(input_image, ((byte*)ret.m_pData) + (yy * ret.m_Stride), yy);
				TIFFClose(input_image);
				return ret;
			}
			#endif

			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<UInt16>::Save(const std::string& filename)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				ToHost(false);
				Synchronize();
				TIFF* output_image = TIFFOpen(filename.c_str(), "w");
				TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, m_Width);
				TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, m_Height);
				TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 16);
				TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
				for (uint32 yy = 0; yy < (uint32)m_Height; yy++) TIFFWriteScanline(output_image, (void*)GetHostScanlinePtr(yy), yy);
				TIFFClose(output_image);
				return;
			}
			#endif

			SaveGeneric(filename);
		}

		#pragma endregion

		#pragma region "Image<float> Load/Save"

		inline /*static*/ Image<float> Image<float>::Load(const string& filename, GPUStream Stream)
		{
			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<float>::Save(const std::string& filename)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			if (IsEqualNoCase(ext, ".raw"))
			{
				wb::io::FileStream fs(filename.c_str(), wb::io::FileMode::Create);
				Save(fs, FileFormat::RAW);
			}
			
			SaveGeneric(filename);
		}

		inline void Image<float>::Save(wb::io::Stream& Stream, FileFormat Format)
		{
			switch (Format)
			{
				/** For BMP support, first convert image to Image<byte>, Image<RGBPixel>, or Image<RGBAPixel> and then Save(). **/

				case FileFormat::RAW:
				{
					/*
					UInt32 nWidth = Width(), nHeight = Height(), nBPP = 64, nStride = m_Stride;
					Stream.Write(&nWidth, sizeof(UInt32));
					Stream.Write(&nHeight, sizeof(UInt32));
					Stream.Write(&nBPP, sizeof(UInt32));
					Stream.Write(&nStride, sizeof(UInt32));
					*/
					ToHost(false);
					Synchronize();
					for (int yy = 0; yy < m_HostData.m_Height; yy++) Stream.Write(GetHostScanlinePtr(yy), m_HostData.m_Width * sizeof(float));
					return;
				}

				default: throw NotSupportedException("Unsupported file format.");
			}
		}

		#pragma endregion

		#pragma region "Image<double> Load/Save"

		inline /*static*/ Image<double> Image<double>::Load(const std::string& filename, GPUStream Stream)
		{
			if (!io::File::Exists(filename.c_str())) throw FileNotFoundException("Image file not found.");
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			#ifdef FITS_Support
			if (IsEqualNoCase(ext, ".fit") || IsEqualNoCase(ext, ".fits"))
			{
				int status = 0;
				fitsfile* pFile;

				fits_open_image(&pFile, filename.c_str(), READONLY, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to open FITS file: ") + err_text).c_str()); }

				int bitpix;
				fits_get_img_type(pFile, &bitpix, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to read FITS file: ") + err_text).c_str()); }
				if (bitpix != DOUBLE_IMG) throw std::exception("Unable to read FITS image: not expected datatype.");

				int naxis;
				fits_get_img_dim(pFile, &naxis, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to read FITS file: ") + err_text).c_str()); }
				if (naxis != 2) throw std::exception("Unable to read FITS image: expected 2D image.");

				long dimensions[2];
				fits_get_img_size(pFile, 2, dimensions, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to read FITS image: ") + err_text).c_str()); }

				long fpixel[2] = { 1, 1 };		// Start from beginning of image
				Image<double> ret(dimensions[0], dimensions[1]);
				fits_read_pix(pFile, bitpix, fpixel, dimensions[0] * dimensions[1], nullptr, ret.BaseAddress(), nullptr, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to read FITS image: ") + err_text).c_str()); }

				fits_close_file(pFile, &status);
				if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to close FITS image: ") + err_text).c_str()); }

				return ret;
			}
			#endif		

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				//TIFFSetWarningHandler(OnTIFFWarning);
				TIFFSetWarningHandler(nullptr);					// Disable warnings.

				TIFF* input_image = TIFFOpen(filename.c_str(), "r");
				uint32 Width, Height;
				uint32 BitsPerSample, SamplesPerPixel;
				uint32 SampleFormat;
				TIFFGetField(input_image, TIFFTAG_IMAGEWIDTH, &Width);
				TIFFGetField(input_image, TIFFTAG_IMAGELENGTH, &Height);
				TIFFGetField(input_image, TIFFTAG_BITSPERSAMPLE, &BitsPerSample);
				TIFFGetField(input_image, TIFFTAG_SAMPLESPERPIXEL, &SamplesPerPixel);
				TIFFGetField(input_image, TIFFTAG_SAMPLEFORMAT, &SampleFormat);
				if (SamplesPerPixel != 1 || SampleFormat != SAMPLEFORMAT_IEEEFP)
					throw std::exception("Expected double-precision floating-point format TIFF image format.");
				Image ret((int)Width, (int)Height);
				for (uint32 yy = 0; yy < Height; yy++) TIFFReadScanline(input_image, ((byte*)ret.m_pData) + (yy * ret.m_Stride), yy);
				TIFFClose(input_image);
				return ret;
			}
			#endif			

			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<double>::Save(const std::string& filename)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			if (IsEqualNoCase(ext, ".raw"))
			{
				wb::io::FileStream fs(filename.c_str(), wb::io::FileMode::Create);
				Save(fs, FileFormat::RAW);
			}

			#ifdef FITS_Support
			if (IsEqualNoCase(ext, ".fit") || IsEqualNoCase(ext, ".fits"))
			{
				wb::io::FileStream fs(pszFilename, wb::io::FileMode::Create);
				Save(fs, FileFormat::FITS);
				return;
			}
			#endif		

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				TIFF* output_image = TIFFOpen(filename.c_str(), "w");
				TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, m_Width);
				TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, m_Height);
				TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 32);
				TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 1);
				TIFFSetField(output_image, TIFFTAG_SAMPLEFORMAT, SAMPLEFORMAT_IEEEFP);
				for (uint32 yy = 0; yy < (uint32)m_Height; yy++) TIFFWriteScanline(output_image, (void*)GetScanlinePtr(yy), yy);
				TIFFClose(output_image);
				return;
			}
			#endif

			SaveGeneric(filename);
		}

		inline void Image<double>::Save(wb::io::Stream& Stream, FileFormat Format)
		{
			switch (Format)
			{
				/** For BMP support, first convert image to Image<byte>, Image<RGBPixel>, or Image<RGBAPixel> and then Save(). **/

				#ifdef FITS_Support
				case FileFormat::FITS:
				{
					int status = 0;
					fitsfile* pFile;

					fits_create_file(&pFile, filename.c_str(), &status);
					if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to create FITS file: ") + err_text).c_str()); }

					long dimensions[2] = { m_Width, m_Height };
					fits_create_img(pFile, DOUBLE_IMG, 2, dimensions, &status);
					if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to create FITS file: ") + err_text).c_str()); }

					long fpixel[2] = { 1, 1 };		// Start from beginning of image
					fits_write_pix(pFile, DOUBLE_IMG, fpixel, m_Width * m_Height, m_pData, &status);
					if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to write FITS file: ") + err_text).c_str()); }

					fits_close_file(pFile, &status);
					if (status != 0) { char err_text[30]; fits_get_errstatus(status, err_text); throw std::exception((std::string("Unable to close FITS image: ") + err_text).c_str()); }

					return;
				}
				#endif

				case FileFormat::RAW:
				{
					/*
					UInt32 nWidth = Width(), nHeight = Height(), nBPP = 64, nStride = m_Stride;
					Stream.Write(&nWidth, sizeof(UInt32));
					Stream.Write(&nHeight, sizeof(UInt32));
					Stream.Write(&nBPP, sizeof(UInt32));
					Stream.Write(&nStride, sizeof(UInt32));
					*/
					for (int yy = 0; yy < m_HostData.m_Height; yy++) Stream.Write(GetHostScanlinePtr(yy), m_HostData.m_Width * sizeof(double));
					return;
				}

				default: throw NotSupportedException("Unsupported file format.");
			}
		}

		#pragma endregion

		#pragma region "Image<RGBPixel> Load/Save"

		inline /*static*/ Image<RGBPixel> Image<RGBPixel>::Load(const string& filename, GPUStream Stream)
		{
			#ifdef FreeImage_Support
			#ifdef LITTLE_ENDIAN
			return FI::ColorLoadHelper<RGBPixel, FIC_RGB, 0x00FF0000, 0x0000FF00, 0x000000FF>(filename, Stream);
			#else
			return FI::ColorLoadHelper<RGBPixel, FIC_RGB, 0x000000FF, 0x0000FF00, 0x00FF0000>(filename, Stream);
			#endif
			#endif

			throw FormatException("Unsupported file format.");
		}

		inline void Image<RGBPixel>::Save(const string& filename)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			#ifndef FreeImage_Support
			if (IsEqualNoCase(ext, ".bmp"))
			{
				wb::io::FileStream fs(filename, wb::io::FileMode::Create);
				Save(fs, FileFormat::BMP);
				return;
			}
			#endif

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				TIFF* output_image = TIFFOpen(pszFilename, "w");
				TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, m_Width);
				TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, m_Height);
				TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
				TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 3);
				TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
				//TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
				TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
				for (uint32 yy = 0; yy < (uint32)m_Height; yy++) TIFFWriteScanline(output_image, (void*)GetScanlinePtr(yy), yy);
				// TIFFWriteEncodedStrip(output_image, 0, &m_image_data[0], m_width*m_height * 4);
				TIFFClose(output_image);
				return;
			}
			#endif

			SaveGeneric(filename);
		}

		inline void Image<RGBPixel>::Save(wb::io::Stream& Stream, FileFormat Format)
		{
			switch (Format)
			{
				case FileFormat::BMP:
				{
					/** Good resources:
					http://tipsandtricks.runicsoft.com/Cpp/BitmapTutorial.html
					http://stackoverflow.com/questions/3142349/drawing-on-8bpp-grayscale-bitmap-unmanaged-c
					**/

					ToHost(false);
					Synchronize();
					Int32 ImageSize = m_HostData.m_Stride * m_HostData.m_Height;

					BITMAPFILEHEADER bmfh;
					BITMAPINFOHEADER info;
					memset(&bmfh, 0, sizeof(BITMAPFILEHEADER));
					memset(&info, 0, sizeof(BITMAPINFOHEADER));

					bmfh.bfType = 0x4d42;       // 0x4d42 = 'BM'			
					bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
					bmfh.bfSize = bmfh.bfOffBits + ImageSize;

					info.biSize = sizeof(BITMAPINFOHEADER);
					info.biWidth = m_HostData.m_Width;
					info.biHeight = m_HostData.m_Height;
					info.biPlanes = 1;
					info.biBitCount = 24;
					info.biCompression = BI_RGB;
					info.biSizeImage = 0;
					info.biXPelsPerMeter = 0x0ec4;
					info.biYPelsPerMeter = 0x0ec4;
					info.biClrUsed = 0;
					info.biClrImportant = 0;

					Stream.Write(&bmfh, sizeof(BITMAPFILEHEADER));
					Stream.Write(&info, sizeof(BITMAPINFOHEADER));
					for (int yy = Height() - 1; yy >= 0; yy--) Stream.Write(GetHostScanlinePtr(yy), m_HostData.m_Stride);
					break;
				}

				#ifdef TIFF_Support        
				case FileFormat::TIFF:
				{
					throw NotImplementedException("TIFF save to stream is not implemented, use filename instead.");
				}
				#endif

				default: throw NotSupportedException("Unsupported file format.");
			}
		}

		#pragma endregion

		#pragma region "Image<RGBAPixel> Load/Save"

		inline /*static*/ Image<RGBAPixel> Image<RGBAPixel>::Load(const string& filename, GPUStream Stream)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			if (IsEqualNoCase(ext, ".bmp"))
			{
				wb::io::FileStream fs(filename, wb::io::FileMode::Open);
				return Load(fs, FileFormat::BMP);
			}

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				Image<RGBAPixel> ret;
				TIFF* tif = TIFFOpen(pszFilename, "r");
				if (tif == nullptr) throw FormatException();
				try
				{
					uint32 w, h;
					TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
					TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
					uint32* raster = (uint32*)_TIFFmalloc(w * h * sizeof(uint32));
					if (raster == nullptr) throw OutOfMemoryException();
					try
					{
						if (!TIFFReadRGBAImage(tif, w, h, raster, 0)) throw IOException();
						ret.Allocate(w, h);
						// Flip vertically and copy
						for (uint32 yy = 0; yy < h; yy++)
							for (uint32 xx = 0; xx < w; xx++)
								ret(xx, ret.Height() - yy - 1) = *(((RGBAPixel*)raster) + (yy * w) + xx);
					}
					catch (std::exception&) { _TIFFfree(raster); throw; }
					_TIFFfree(raster);
				}
				catch (std::exception&) { TIFFClose(tif); throw; }
				TIFFClose(tif);
				return ret;
			}
			#endif

			#ifdef FreeImage_Support
			#ifdef LITTLE_ENDIAN
			return FI::ColorLoadHelper<RGBAPixel, FIC_RGBALPHA, 0x00FF0000, 0x0000FF00, 0x000000FF>(filename, Stream);
			#else
			return FI::ColorLoadHelper<RGBAPixel, FIC_RGBALPHA, 0x000000FF, 0x0000FF00, 0x00FF0000>(filename, Stream);
			#endif
			#endif

			throw FormatException("Unsupported file format.");			
		}

		inline /*static*/ Image<RGBAPixel> Image<RGBAPixel>::Load(wb::io::Stream& Stream, FileFormat Format, GPUStream gpuStream)
		{
			switch (Format)
			{
				case FileFormat::BMP:
				{
					BITMAPFILEHEADER bmfh;
					BITMAPINFOHEADER info;
					memset(&bmfh, 0, sizeof(BITMAPFILEHEADER));
					memset(&info, 0, sizeof(BITMAPINFOHEADER));

					Stream.Read(&bmfh, sizeof(BITMAPFILEHEADER));
					if (bmfh.bfType != 0x4d42) throw FormatException("Not a valid bitmap image file.");
					Stream.Read(&info, sizeof(BITMAPINFOHEADER));
					if (info.biSize != sizeof(BITMAPINFOHEADER)) throw FormatException("Not a valid bitmap image file.");
					if (info.biCompression != BI_RGB) throw NotSupportedException("Only uncompressed bitmaps are supported.");
					if ((info.biBitCount != 24 && info.biBitCount != 32) || info.biClrUsed != 0)
						throw NotSupportedException("This routine only supports 24/32-bit per pixel bitmaps.");
					//Stream.Seek(bmfh.bfOffBits, wb::io::SeekOrigin::Begin);
					int Height = abs(info.biHeight);

					// See "Calculating Surface Stride" here:  https://docs.microsoft.com/en-us/windows/win32/api/wingdi/ns-wingdi-bitmapinfoheader
					uint FileStride = ((((info.biWidth * info.biBitCount) + 31) & ~31) >> 3);

					if (info.biBitCount == 24)
					{
						auto dst = Image<RGBPixel>::NewHostImage(info.biWidth, Height, gpuStream);
						auto Pos = Stream.GetPosition();
						for (int yy = dst.Height() - 1; yy >= 0; yy--)
						{
							Stream.Read(dst.GetHostScanlinePtr(yy), dst.m_HostData.m_Width * sizeof(RGBPixel));
							Pos += FileStride;
							Stream.Seek(Pos, wb::io::SeekOrigin::Begin);
						}
						auto ret = Image<RGBAPixel>::NewHostImage(dst.Width(), dst.Height(), gpuStream);
						dst.ConvertTo(ret);
						return ret;
					}
					else
					{
						auto ret = Image<RGBAPixel>::NewHostImage(info.biWidth, Height, gpuStream);
						for (int yy = ret.Height() - 1; yy >= 0; yy--)
							Stream.Read(ret.GetHostScanlinePtr(yy), ret.m_HostData.m_Stride);
						return ret;
					}
				}

				default: throw FormatException("Unsupported file format.");
			}
		}

		inline void Image<RGBAPixel>::Save(const string& filename)
		{
			string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filename)));

			#ifndef FreeImage_Support
			if (IsEqualNoCase(ext, ".bmp"))
			{
				wb::io::FileStream fs(filename, wb::io::FileMode::Create);
				Save(fs, FileFormat::BMP);
				return;
			}
			#endif

			#ifdef LibTIFF_Support
			if (IsEqualNoCase(ext, ".tif") || IsEqualNoCase(ext, ".tiff"))
			{
				TIFF* output_image = TIFFOpen(pszFilename, "w");
				TIFFSetField(output_image, TIFFTAG_IMAGEWIDTH, m_Width);
				TIFFSetField(output_image, TIFFTAG_IMAGELENGTH, m_Height);
				TIFFSetField(output_image, TIFFTAG_BITSPERSAMPLE, 8);
				TIFFSetField(output_image, TIFFTAG_SAMPLESPERPIXEL, 4);
				TIFFSetField(output_image, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
				//TIFFSetField(output_image, TIFFTAG_COMPRESSION, COMPRESSION_DEFLATE);
				TIFFSetField(output_image, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_RGB);
				for (uint32 yy = 0; yy < (uint32)m_Height; yy++) TIFFWriteScanline(output_image, (void*)GetScanlinePtr(yy), yy);
				// TIFFWriteEncodedStrip(output_image, 0, &m_image_data[0], m_width*m_height * 4);
				TIFFClose(output_image);
				return;
			}
			#endif

			SaveGeneric(filename);
		}

		inline void Image<RGBAPixel>::Save(wb::io::Stream& Stream, FileFormat Format)
		{
			switch (Format)
			{
				case FileFormat::BMP:
				{
					/** Good resources:
					http://tipsandtricks.runicsoft.com/Cpp/BitmapTutorial.html
					http://stackoverflow.com/questions/3142349/drawing-on-8bpp-grayscale-bitmap-unmanaged-c
					**/

					ToHost(false);
					Synchronize();
					Int32 ImageSize = m_HostData.m_Stride * m_HostData.m_Height;

					BITMAPFILEHEADER bmfh;
					BITMAPINFOHEADER info;
					memset(&bmfh, 0, sizeof(BITMAPFILEHEADER));
					memset(&info, 0, sizeof(BITMAPINFOHEADER));

					bmfh.bfType = 0x4d42;       // 0x4d42 = 'BM'			
					bmfh.bfOffBits = sizeof(BITMAPFILEHEADER) + sizeof(BITMAPINFOHEADER);
					bmfh.bfSize = bmfh.bfOffBits + ImageSize;

					info.biSize = sizeof(BITMAPINFOHEADER);
					info.biWidth = m_HostData.m_Width;
					info.biHeight = m_HostData.m_Height;
					info.biPlanes = 1;
					info.biBitCount = 32;
					info.biCompression = BI_RGB;
					info.biSizeImage = 0;
					info.biXPelsPerMeter = 0x0ec4;
					info.biYPelsPerMeter = 0x0ec4;
					info.biClrUsed = 0;
					info.biClrImportant = 0;

					Stream.Write(&bmfh, sizeof(BITMAPFILEHEADER));
					Stream.Write(&info, sizeof(BITMAPINFOHEADER));
					for (int yy = Height() - 1; yy >= 0; yy--) Stream.Write(GetHostScanlinePtr(yy), m_HostData.m_Stride);
					break;
				}

				#ifdef TIFF_Support        
				case FileFormat::TIFF:
				{
					throw NotSupportedException("TIFF is currently only supported via filename save.");
				}
				#endif

				default: throw NotSupportedException("Unsupported file format.");
			}
		}

		#pragma endregion

		#pragma region "Image<thurst::complex<float>> and Image<thrust::complex<double>> Load/Save"

		inline /*static*/ Image<thrust::complex<float>> Image<thrust::complex<float>>::Load(const std::string& filename, GPUStream Stream) {
			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<thrust::complex<float>>::Save(const string& filename) {
			SaveGeneric(filename);
		}

		inline /*static*/ Image<thrust::complex<double>> Image<thrust::complex<double>>::Load(const std::string& filename, GPUStream Stream) {
			return base::LoadGeneric(filename, Stream);
		}

		inline void Image<thrust::complex<double>>::Save(const string& filename) {
			SaveGeneric(filename);
		}

		#pragma endregion

	}// namespace images
}// namespace wb

#endif	// __WBImages_LoadSave_h__

//	End of Images_LoadSave.h


