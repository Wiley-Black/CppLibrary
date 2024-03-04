/////////
//	Images_Base.h
//
///
//	Design Notes:
/*
		Programming patterns:

			In image processing routines, I see a few categories that I could potentially place operations in:
				* operation "to" an image.  I.e. convert a starting double image "to" a single float image.
				* operation "from" an image.  I.e. convert a single float image "from" a starting double image.
				* operation "in-place".  I.e. saturate pixels in an image.

		Looking at other well-thought out libraries:
			- numpy has a Pillow .convert() function that is a member of the class you are converting from.  It accepts an argument that specifies what type to go to, which is
				more suitable to the soft typing of Python.
			- MATLAB has the rgb2gray() function that is not a member of anything.
			- OpenCV has a cv::cvtColor() routine that takes in a source and a destination image.  This is a colorspace conversion function, so in the category of ToGrayscale().

		From the caller perspective, I think the "to" programming pattern is more readible.  The pattern varies a 
		little depending on exactly the operation, but for example compare how the following are written:

			// "to":
			auto result = StartingImage.OperationTo<TargetPixelType>();

			// "from":
			auto result = TargetType::OperationFrom(StartingImage);

			// "in-place":
			StartingImage.OperationInPlace();

		Both to and from have some downsides.  The "to" approach may require specifying a template parameter.  The
		"from" approach may still need a type specified, but now it goes on the left as the class we're coming
		"into".  I think for the caller the "to" approach is ultimately more readable and flows better- you
		start by writing the image you are working with, and then a dot, and then what you want done to it.  Also,
		the "to" approach allows stringing of operations:

			auto final_result = StartingImage.Operation_1_To().Operation_2_To().Operation_3_To();

		For a more concrete example:

			auto final_result = StartingImage.ResizeTo(512, 512).ConvertTo<float>().Saturate(0.0f, 1.0f);

		In both the "to" and "from" cases, there is the possibility of an operation being performed on an existing
		image buffer or a brand new buffer.  For example, the above example creates 3 new images- a resized image,
		a converted image, and a saturated image.  In a pipeline with pre-allocated and reused image buffers,
		you might instead want to re-use buffers.  The above could be rewritten:

			auto final_result = StartingImage.ResizeTo(BufferA, 512, 512).ConvertTo<float>(BufferB).SaturateTo(BufferC, 0.0f, 1.0f);

		A subtle distinction between the new and existing buffers is that the return type would transition from being an
		Image<float> to being an Image<float>&, a reference to BufferC in the end.

		This programming pattern allows some functions to be written as an in-place version without much difference in caller
		structure:

			auto final_result = StartingImage.ResizeTo(BufferA, 512, 512).ConvertTo<float>(BufferB).SaturateInPlace(0.0f, 1.0f);

		This last example above now returns a reference to BufferB, after the saturate-in-place operation has been performed.
		The pattern also supports mixing of new and existing buffers.  For example:

			auto final_result = StartingImage.ResizeTo(512, 512).ConvertTo<float>(BufferB).SaturateInPlace(0.0f, 1.0f);

		This example creates a new buffer for resizing but recycles BufferB and finally applies the saturate on BufferB in-place.		
*/
/////////

#ifndef __WBImages_Base_h__
#define __WBImages_Base_h__

#include "../../wbFoundation.h"
#include "../../wbCore.h"

#include "../../System/GPU.h"			// Includes cuda.h, if supported.  Defines empty but convenient GPUStream if not supported.
#include "PixelType_Primitives.h"
#include "Images_Memory.h"
#include "Images_Kernels.h"

namespace wb
{
	namespace images
	{
		/** References **/
		using namespace wb::cuda;
		template<typename PixelType> class ConvolutionKernel;				// Forward declaration.

		#pragma region "Helpers"

		#define AfterKernelLaunch()		AfterKernelLaunchAux(__FILE__, __LINE__)		

		#pragma endregion

		#pragma region "Rectangle Primitive"

		template<typename OrdinateType> struct Rectangle
		{
			OrdinateType X, Y;
			OrdinateType Width, Height;

			Rectangle() { }
			Rectangle(OrdinateType _X, OrdinateType _Y, OrdinateType _Width, OrdinateType _Height)
			{
				X = _X; Y = _Y; Width = _Width; Height = _Height;
			}

			// Inclusive:
			OrdinateType Left() const { return X; }
			OrdinateType Top() const { return Y; }

			void SetLeft(OrdinateType value) { Width += (X - value); X = value; }
			void SetTop(OrdinateType value) { Height += (Y - value); Y = value; }

			// Exclusive:
			OrdinateType Right() const { return X + Width; }
			OrdinateType Bottom() const { return Y + Height; }

			void SetRight(OrdinateType value) { Width = value - X; }
			void SetBottom(OrdinateType value) { Height = value - Y; }

			OrdinateType Area() const { return Width * Height; }

			// Special Values
			static const Rectangle Whole() { return Rectangle(0, 0, std::numeric_limits<OrdinateType>::max(), std::numeric_limits<OrdinateType>::max()); }
			bool IsWhole() const { return *this == Whole(); }

			// Operations
			void ConstrainTo(const Rectangle<OrdinateType>& MaxRegion)
			{
				if (X < MaxRegion.X) X = MaxRegion.X; else if (X >= MaxRegion.Right()) X = MaxRegion.Right() - 1;
				if (Y < MaxRegion.Y) Y = MaxRegion.Y; else if (Y >= MaxRegion.Bottom()) Y = MaxRegion.Bottom() - 1;
				if (Width < 0) Width = 0;
				if (Height < 0) Height = 0;
				if (Right() > MaxRegion.Right()) Width = MaxRegion.Right() - X;
				if (Bottom() > MaxRegion.Bottom()) Height = MaxRegion.Bottom() - Y;
			}

			/// <summary>IsContainedIn() returns true if this rectangle is fully contained within Container.  A rectangle that
			/// matches its container edges returns true (i.e. Rectangle X=0, Width=10 IsContainedIn(Rectangle X=0, Width=10)
			/// is true).</summary>
			bool IsContainedIn(const Rectangle<OrdinateType>& Container) const
			{
				return (X >= Container.X && Y >= Container.Y && Right() <= Container.Right() && Bottom() <= Container.Bottom());
			}

			bool operator==(const Rectangle<OrdinateType>& rhs) const { return (X == rhs.X && Y == rhs.Y && Width == rhs.Width && Height == rhs.Height); }
			bool operator!=(const Rectangle<OrdinateType>& rhs) const { return (X != rhs.X || Y != rhs.Y || Width != rhs.Width || Height != rhs.Height); }

			#ifdef NPP_Support
			operator NppiPoint() const { NppiPoint ret; ret.x = X; ret.y = Y; return ret; }
			operator NppiSize() const { NppiSize ret; ret.width = Width; ret.height = Height; return ret; }
			operator NppiRect() const { NppiRect ret; ret.x = X; ret.y = Y; ret.width = Width; ret.height = Height; return ret; }
			#endif
		};

		#pragma endregion

		#pragma region "Dynamic Range"

		// Declare the Range type a.k.a. as Image<float>::Range and Image<double>::Range.  Have to do this
		// because C++ does not allow forward declarations of nested types.  The "real" class will be
		// Image_Range, so that it can be forward-declared, but typedefs inside of Image allows us to
		// treat the class as if it is Image<float>::Range, etc.
		template<typename PixelType> struct Image_Range
		{
			PixelType Minimum;
			PixelType Maximum;

			static const Image_Range Unit;

			Image_Range() { Minimum = 0.0; Maximum = 1.0; }
			Image_Range(const Image_Range& cp) : Minimum(cp.Minimum), Maximum(cp.Maximum) { }
			Image_Range(Image_Range&& mv) : Minimum(mv.Minimum), Maximum(mv.Maximum) { }
			Image_Range(PixelType _Minimum, PixelType _Maximum) : Minimum(_Minimum), Maximum(_Maximum) { }
			Image_Range& operator=(const Image_Range& cp) { Minimum = cp.Minimum; Maximum = cp.Maximum; return *this; }
			Image_Range& operator=(Image_Range&& mv) { Minimum = mv.Minimum; Maximum = mv.Maximum; return *this; }
		};

		#pragma endregion				

		#pragma region "NVIDIA Performance Primitives (NPP) Library Interface"

		enum class InterpolationMethods
		{
			NearestNeighbor = NPPI_INTER_NN,
			Linear = NPPI_INTER_LINEAR,
			Cubic = NPPI_INTER_CUBIC,
			Super = NPPI_INTER_SUPER,
			Lanczos = NPPI_INTER_LANCZOS
		};

		// NPP = NVIDIA Performance Primitives
		// NPPI = NVIDIA Performance Primitives, Imaging

		#ifdef NPP_Support		
		namespace NPPI
		{
			template<typename PixelType, typename KernelType> struct FilterBehaviors
			{
				// Undefined by default- must define specializations or an error will be generated upon attempt to use NPPI routines.			
			};

			template<typename KernelType> struct KernelBehaviors
			{
				// Undefined by default- must define specializations or an error will be generated upon attempt to use NPPI routines.			
			};

			template<typename PixelType> struct Behaviors
			{
				// Undefined by default- must define specializations or an error will be generated upon attempt to use NPPI routines.			
			};			

			class Size : public NppiSize
			{
			public:
				Size() { width = 0; height = 0; }
				Size(int w, int h) { width = w; height = h; }
			};

			class Point : public NppiPoint
			{
			public:
				Point() { x = y = 0; }
				Point(int x, int y) { this->x = x; this->y = y; }
			};

			class Rect : public NppiRect
			{
			public:
				Rect() { x = y = width = height = 0; }
				Rect(int x, int y, int width, int height) { this->x = x; this->y = y; this->width = width; this->height = height; }
				Rect(const Rectangle<int>& cp) { this->x = cp.X; this->y = cp.Y; this->width = cp.Width; this->height = cp.Height; }
				operator Rectangle<int>() const { return Rectangle<int>(x, y, width, height); }
			};

		};	// End namespace NPPI
		#endif
		#pragma endregion

		#pragma region "Image Foundations"

		template<typename PixelType, typename FinalType> class BaseImage : public wb::diagnostics::DebugValidObject
		{
			typedef DebugValidObject base;

		public:
			#pragma region "Declarations"

			// Allow Image_Range<T> to also be written as Image<T>::Range for most purposes:
			typedef Image_Range<PixelType> Range;

			#pragma endregion

			#pragma region "Internal"

		public:
			typedef memory::HostImageData<PixelType> HostImageData;
			#ifdef CUDA_Support
			typedef memory::DeviceImageData<PixelType> DeviceImageData;
			#endif
			typedef memory::DataState DataState;

			DataState			m_DataState;		// Which ImageData is up-to-date?  Or both/neither?			
			HostImageData		m_HostData;
			#ifdef CUDA_Support
			DeviceImageData		m_DeviceData;
			#endif

			// Many cuda operations happen asynchronously.  If this image has been touched by one,
			// then m_bPending is true until a Synchronize() operation is performed on the m_Pending stream.
			// If m_bPending is true, then an attempt to perform an operation on the data from a different
			// cudaStream_t must also be blocked until synchronization.  This can apply to either the
			// host or device data, since an asynchronous operation could apply in either.
			bool				m_bPending;
			cudaStream_t		m_Pending;						

		protected:
			GPUStream			m_Stream;			// Note: when CUDA_Support is absent, just an empty placeholder class.			

			/// <summary>
			/// In the case where both host and device are up-to-date, which was used
			/// more recently?  i.e. After the ToDevice() call, TowardHost goes to false
			/// because the caller is moving toward the device.  As soon as an operation
			/// must pare off and edit the image, TowardHost can select the direction
			/// to go.
			/// </summary>
			bool				m_TowardHost;

			DataState& GetDataState() { return m_DataState; }
			HostImageData& GetHostData() { return m_HostData; }
			#ifdef CUDA_Support
			DeviceImageData& GetDeviceData() { return m_DeviceData; }
			#endif

			template<typename PixelTypeDst, typename FinalTypeDst> FinalTypeDst& TypecastConvertTo(FinalTypeDst& dst);

		public:

			static bool WouldModifyInHost(DataState State, bool TowardHost)
			{
				return (State == DataState::Host ||
					(State == DataState::HostAndDevice && TowardHost));
			}

			bool WouldModifyInHost() {
				ValidateObject();
				return WouldModifyInHost(m_DataState, m_TowardHost);
			}

			template<typename PixelType2, typename FinalType2> bool WouldModifyInHost(BaseImage<PixelType2, FinalType2>& WithReadSource) {
				ValidateObject();

				// The easiest case: if the read data is available in both places, then we
				// can base our decision entirely on 'this' and revert to simple WouldModifyInHost().
				if (WithReadSource.m_DataState == DataState::HostAndDevice) return WouldModifyInHost();

				// Another simple case: if this side is available in HostAndDevice but the
				// other image is in a specific place, it is much faster to skip the memory
				// move and go where the data already is.  We'll handle this as overriding the
				// TowardHost flag locally for this routine.  The only case where the original
				// m_TowardHost flag would have been used is in the above case where we've 
				// already branched off to ModifyInHost().
				bool TowardHost = (WithReadSource.m_DataState == DataState::Host);

				return (m_DataState == DataState::Host ||
					(m_DataState == DataState::HostAndDevice && TowardHost));
			}

			/// <summary>
			/// ModifyInHost() is a helper function that supports the following programming
			/// pattern:
			/// 
			///		if (ModifyInHost())
			///			// Can now change the host data.
			///		else
			///			// Can now change the device data.
			/// 
			/// ModifyInHost() is phrased as a preference for the host-side, but it 
			/// applies no bias between choosing the host or device path except where 
			/// the data currently sits or was most recently accessed.  
			/// 
			/// After the ModifyInHost() call, it is assumed that the data *will* be 
			/// modified according to the pattern above.  That is, after the call, the
			/// data is marked as up-to-date only in either the Host or the Device and
			/// 
			/// It is responsible for ascertaining the current status of the data in this
			/// image and, in the "true" or "host" case, ensuring that the host data is
			/// synchronized.  In the event that the data is up-to-date in both host and
			/// device, the m_TowardHost flag is used to proceed along whatever the most
			/// recent host/device operation was.  ModifyInHost() also checks that
			/// the state is not DataState::None and throws an exception if it is- the
			/// ModifyInHost() function should be used when a modification is called
			/// for and allocation is already expected.  												
			/// not both.
			/// </summary>
			/// <returns>True if modification of the host data can proceed now.  False if
			/// the modification should be to the device data.</returns>
			bool	ModifyInHost()
			{
				ValidateObject();

				if (WouldModifyInHost(m_DataState, m_TowardHost))
				{
					Synchronize();
					m_DataState = DataState::Host;
					return true;
				}
				else
				{
					if (m_DataState == DataState::None) throw NotSupportedException();
					m_DataState = DataState::Device;
					return false;
				}
			}

			/// <summary>
			/// This variation of ModifyInHost() works the same as the base ModifyInHost(),
			/// but considers the current state of a second image that will be a read
			/// source for the operation.  It also ensures that both images are available 
			/// in the returned state so that they may be read/accessed- i.e. if the returned 
			/// answer is "false", then both images are assured as being in either the Device 
			/// or HostAndDevice states.  Since the 2nd image is to be read-only for this 
			/// operation, it is not transitioned out of the HostAndDevice state.
			/// </summary>
			template<typename PixelType2, typename FinalType2> bool	ModifyInHost(BaseImage<PixelType2, FinalType2>& WithReadSource)
			{
				ValidateObject();

				if (WouldModifyInHost(WithReadSource))
				{
					Synchronize();
					m_DataState = DataState::Host;
					WithReadSource.ToHost();		// Has no effect if already on the host or in both.
					WithReadSource.Synchronize();
					return true;
				}
				else
				{
					if (m_DataState == DataState::None) throw NotSupportedException();
					m_DataState = DataState::Device;
					WithReadSource.ToDevice();		// Has no effect if already on the device or in both.
					return false;
				}
			}			

			#pragma endregion
		
			#pragma region "File Formats"

			enum class FileFormat
			{
				Unspecified,
				Unrecognized,
				BMP,
				FITS,
				RAW,
				TIFF
			};

			static string to_string(FileFormat ff)
			{
				switch (ff) {					
				case FileFormat::Unspecified: return "Unspecified";				
				case FileFormat::BMP: return "BMP";
				default:
				case FileFormat::Unrecognized: return "Unrecognized";
				}
			}

			static FileFormat ToFileFormat(const wb::io::Path& filepath)
			{
				string ext = to_lower(wb::to_string(wb::io::Path::GetExtension(filepath)));
				if (ext == ".bmp") return FileFormat::BMP;
				return FileFormat::Unrecognized;
			}

			static FinalType LoadGeneric(const std::string& filename, GPUStream Stream = GPUStream::None());
			void SaveGeneric(const std::string& filename, bool ApplyCompression = true);

			#pragma endregion		

			#pragma region "Construction and Host-Device Management"
		protected:

			BaseImage(GPUStream Stream, HostFlags HostFlags)
				: m_Stream(Stream), m_HostData(HostFlags), m_Pending((cudaStream_t)0), m_bPending(false)
			{
				m_DataState = DataState::None;
				m_TowardHost = true;
			}

		public:

			BaseImage()
				: m_Stream(GPUStream::None()), m_Pending((cudaStream_t)0), m_bPending(false)
			{
			}

			BaseImage(BaseImage&) = delete;
			BaseImage& operator=(BaseImage&) = delete;
			
			BaseImage(BaseImage<PixelType, FinalType>&& mv) noexcept
				: base(std::move(mv)), m_Stream(std::move(mv.m_Stream)),
				m_HostData(std::move(mv.m_HostData))
				#ifdef CUDA_Support
				, m_DeviceData(std::move(mv.m_DeviceData))
				#endif
			{
				m_DataState = mv.m_DataState; mv.m_DataState = DataState::None;
				m_TowardHost = mv.m_TowardHost;												
				m_bPending = mv.m_bPending; mv.m_bPending = false;
				m_Pending = mv.m_Pending;
			}

			BaseImage& operator=(BaseImage<PixelType, FinalType>&& mv)
			{
				base::operator=(std::move(mv));
				m_Stream = std::move(mv.m_Stream);
				m_DataState = mv.m_DataState; mv.m_DataState = DataState::None;
				m_HostData = std::move(mv.m_HostData);
				#ifdef CUDA_Support
				m_DeviceData = std::move(mv.m_DeviceData);
				#endif
				m_TowardHost = mv.m_TowardHost;
				m_bPending = mv.m_bPending; mv.m_bPending = false;
				m_Pending = mv.m_Pending;
				return *this;
			}

			static FinalType NewHostImage(int Width, int Height, GPUStream Stream = GPUStream::None(), HostFlags Flags = HostFlags::None)
			{				
				FinalType ret(Stream, Flags);
				ret.m_HostData.Allocate(Width, Height, true);
				ret.m_DataState = DataState::Host;
				ret.m_TowardHost = true;
				return ret;
			}

			static FinalType ExistingHostImageWrapper(int Width, int Height, int Stride, PixelType* pData, bool CanWrite, GPUStream Stream = GPUStream::None(), HostFlags flags = HostFlags::None)
			{
				FinalType ret(Stream, HostFlags::None);
				ret.m_HostData = HostImageData::NewWrapper(Width, Height, Stride, pData, CanWrite, flags);
				ret.m_DataState = DataState::Host;
				ret.m_TowardHost = true;
				return ret;
			}

			#ifdef CUDA_Support
			static FinalType NewDeviceImage(int Width, int Height, GPUStream Stream, HostFlags Flags = HostFlags::Pinned)
			{
				FinalType ret(Stream, Flags);
				ret.m_DeviceData.Allocate(Width, Height, true, false, ret.m_bPending, ret.m_Pending, Stream);
				ret.m_DataState = DataState::Device;
				ret.m_TowardHost = false;
				return ret;
			}
			#endif		

			/// <summary>
			/// CopyToHost() will retain the stream from this image as the stream in the
			/// new image.  To switch streams, call SetStream() before or after the copy 
			/// is complete.  The copy happens asynchronously, and a Synchronize() call
			/// should be made before accessing the data.
			/// </summary>
			FinalType CopyToHost(HostFlags flags = HostFlags::Retain)
			{
				if (flags == HostFlags::Retain) flags = m_HostData.GetFlags();

				switch (m_DataState)
				{
				case DataState::None:
				default: throw NotSupportedException("Cannot transfer image to host- no data in source image.");
				case DataState::Host:
				case DataState::HostAndDevice:
				{
					FinalType ret(m_Stream, flags);
					StartAsync(m_Stream);
					ret.StartAsync(m_Stream);
					ret.m_HostData.Allocate(m_HostData.m_Width, m_HostData.m_Height, true);
					m_HostData.AsyncCopyImageTo(ret.m_HostData, m_Stream);
					ret.m_DataState = DataState::Host;
					ret.m_TowardHost = true;
					return ret;
				}
				#ifdef CUDA_Support
				case DataState::Device:
				{
					FinalType ret(m_Stream, flags);
					StartAsync(m_Stream);
					ret.StartAsync(m_Stream);
					ret.m_HostData.Allocate(m_DeviceData.m_Width, m_DeviceData.m_Height, true);
					m_DeviceData.AsyncCopyImageTo(ret.m_HostData, m_Stream);
					ret.m_DataState = DataState::Host;
					ret.m_TowardHost = true;
					return ret;
				}
				#endif
				}
			}

			/// <summary>
			/// CopyToDevice() creates a new image that contains the copied data from this image.  The new image is placed on
			/// the device side only.  If possible, the data is copied device-to-device, but if not then it is copied 
			/// host-to-device.  The copy happens asynchronously.
			/// </summary>
			FinalType CopyToDevice(HostFlags flags = HostFlags::Retain)
			{
				if (flags == HostFlags::Retain) flags = m_HostData.GetFlags();

				switch (m_DataState)
				{
				case DataState::None:
				default: throw NotSupportedException("Cannot transfer image to host- no data in source image.");
				case DataState::Host:
				{
					FinalType ret(m_Stream, flags);
					ret.m_DeviceData.Allocate(m_HostData.m_Width, m_HostData.m_Height, true, false, ret.m_bPending, ret.m_Pending, m_Stream);
					StartAsync(m_Stream);
					ret.StartAsync(m_Stream);
					m_HostData.AsyncCopyImageTo(ret.m_DeviceData, m_Stream);
					ret.m_DataState = DataState::Device;
					ret.m_TowardHost = false;
					return ret;
				}
				#ifdef CUDA_Support
				case DataState::HostAndDevice:
				case DataState::Device:
				{
					FinalType ret(m_Stream, flags);
					ret.m_DeviceData.Allocate(m_DeviceData.m_Width, m_DeviceData.m_Height, true, false, ret.m_bPending, ret.m_Pending, m_Stream);
					StartAsync(m_Stream);
					ret.StartAsync(m_Stream);
					m_DeviceData.AsyncCopyImageTo(ret.m_DeviceData, m_Stream);
					ret.m_DataState = DataState::Device;
					ret.m_TowardHost = false;
					return ret;
				}
				#endif
				}
			}

			/// <summary>
			/// The ToHost() transfers image data to the host-side if it is not already there.
			/// If the NeedWritable flag is true, then the image is marked as valid only on the 
			/// host-side.  Set the NeedWritable flag to false when not writing so as to avoid 
			/// unnecessary image transfers between the host and device.  That is, when the 
			/// NeedWritable flag is false, the image state can be in both locations and further
			/// ToDevice() calls remain instanteous.   
			/// 
			/// Multiple calls to ToHost() are instanteous (if nothing relevant happens in between),
			/// and so it is possible to call ToHost() with the NeedWritable flag false and later 
			/// call again with the flag true to mark that the image has or will be changed in 
			/// the interim.
			/// 
			/// The ToHost() call happens asynchronously when possible.
			/// </summary>
			virtual void ToHost(bool NeedWritable = false)
			{
				ValidateObject();
				m_TowardHost = true;
				switch (m_DataState)
				{
				case DataState::HostAndDevice:
					if (NeedWritable) m_DataState = DataState::Host;
				case DataState::Host:
					return;
				#ifdef CUDA_Support
				case DataState::Device:
					StartAsync(m_Stream);
					m_HostData.Allocate(m_DeviceData.m_Width, m_DeviceData.m_Height, NeedWritable);
					m_DeviceData.AsyncCopyImageTo(m_HostData, m_Stream);
					if (NeedWritable) m_DataState = DataState::Host; else m_DataState = DataState::HostAndDevice;
					return;
				#endif
				default: throw NotSupportedException();
				}
			}

			/// <summary>
			/// The ToDevice() transfers image data to the device-side if it is not already
			/// available.  The NeedWritable flag follows the same mechanics as in ToHost().  ToDevice() 
			/// uses an asynchronous copy/stream if possible and the data cannot be assumed to actually 
			/// be on the device until Synchronize() is called- but the device data can be accessed using the 
			/// image's current stream and CUDA will manage the stream.			
			/// </summary>
			virtual void ToDevice(bool NeedWritable = false)
			{
				#ifdef CUDA_Support
				ValidateObject();
				m_TowardHost = false;
				switch (m_DataState)
				{
				case DataState::Host:
					m_DeviceData.Allocate(m_HostData.m_Width, m_HostData.m_Height, NeedWritable, false, m_bPending, m_Pending, m_Stream);
					StartAsync(m_Stream);
					m_HostData.AsyncCopyImageTo(m_DeviceData, m_Stream);
					if (NeedWritable) m_DataState = DataState::Device; else m_DataState = DataState::HostAndDevice;
					return;
				case DataState::HostAndDevice:
					if (NeedWritable) m_DataState = DataState::Device;
				case DataState::Device:
					return;
				default: throw NotSupportedException();
				}
				#else
				throw NotSupportedException();
				#endif
			}						

			#pragma endregion

			#pragma region "Stream and Synchronization"

			void SetStream(GPUStream Stream)
			{
				// Note: the ImageData classes track pending operations and their stream.
				// The StartAsync() call will force a synchronize if the underlying cudaStream_t
				// on an image is different than the one for a new StartAsync() call.  Because
				// of this lower-level tracking, there is no need to force a synchronization
				// here.
				#ifdef CUDA_Support
				m_Stream = GPUStream(Stream);
				#endif
			}

			/// <summary>
			/// Call StartAsync() just before beginning any new asynchronous operation
			/// on this dataset for the given stream.  If a pending asynchronous operation
			/// from a different stream exists on this stream, then StartAsync() will 
			/// synchronize that pending stream before recording the new operation
			/// starting.
			/// 
			/// Multiple calls to StartAsync() can be chained without calling AfterKernelLaunch()
			/// or any other closure.  If the stream is consistent in all calls, then the only 
			/// effect will be to check for an error code from CUDA.  Calling Synchronize()
			/// will block until the asynchronous operations complete.
			/// </summary>
			/// <param name="NewPending">Stream on which the new operation will start.</param>
			void StartAsync(cudaStream_t NewPending)
			{
				ValidateObject();
				if (!m_bPending && NewPending != m_Pending) Synchronize();
				m_bPending = true;
				m_Pending = NewPending;

				// To check for kernel launch errors, we will require a call to 
				// cudaPeekAtLastError().  However we won't know if it is a launch
				// error from the launch we just made unless we clear any past errors.
				// cudaGetLastError() resets the host thread-based error variable to
				// cudaSuccess.
				cudaThrowable(cudaGetLastError());
			}

			/// <summary>
			/// Call StartAsync() just before beginning any new asynchronous operation
			/// on this dataset for the given stream.  If a pending asynchronous operation
			/// from a different stream exists on this stream, then StartAsync() will 
			/// synchronize that pending stream before recording the new operation
			/// starting.
			/// 
			/// Multiple calls to StartAsync() can be chained without calling AfterKernelLaunch()
			/// or any other closure.  If the stream is consistent in all calls, then the only 
			/// effect will be to check for an error code from CUDA.  Calling Synchronize()
			/// will block until the asynchronous operations complete.
			/// </summary>
			/// <param name="NewPending">Stream on which the new operation will start.</param>
			void StartAsync()
			{
				StartAsync(m_Stream);
			}

			/// <summary>
			/// Synchronize() provides a blocking call that ensures all asynchronous operations on this image's
			/// stream have completed before returning.  If there are no pending operations on the image (i.e.
			/// from a call to StartAsync() or various methods), then Synchronize() returns immediately.
			/// </summary>
			void Synchronize() {
				if (m_bPending)
				{
					#ifdef CUDA_Support
					if (m_Pending == (cudaStream_t)0) throw Exception("Attempting to synchronize on null stream handle.");
					cudaThrowable(cudaStreamSynchronize(m_Pending));
					#endif
					m_bPending = false;
				}
			}

			#pragma endregion

			#pragma region "CUDA Helpers"
			#ifdef CUDA_Support

			/// <summary>
			/// GetSmallOpKernelParameters() provides kernel parameters suitable for a fast, pixelwise operation.
			/// Examples include a single multiplication for each pixel.  
			/// </summary>			
			void GetSmallOpKernelParameters(dim3& blocks, dim3& threads)
			{
				//unsigned int MaxThreadsPerBlock = 32 ? pStream == nullptr : pStream->device_properties.maxThreadsPerBlock;
				unsigned int MaxThreadsPerBlock = m_Stream.GetDeviceProperties().maxThreadsPerBlock;
				//threads = dim3((int)sqrt(MaxThreadsPerBlock), (int)sqrt(MaxThreadsPerBlock));
				const int ThreadsX = 8;
				// Based on profiling, this has a very very small effect, probably within the noise and all choices of
				// X from 2^1 to 2^10 had very similar performance, but 2^3 had a slight minima in two trials.
				threads = dim3(ThreadsX, MaxThreadsPerBlock / ThreadsX, 1);
				blocks = dim3(divup(m_DeviceData.m_Width, threads.x), divup(m_DeviceData.m_Height, threads.y));
			}

			cuda::img_format GetCudaFormat() const { return m_DeviceData.GetCudaFormat(); }				

			#endif
			#pragma endregion

			#pragma region "Properties"

			int Width() const {
				switch (m_DataState) {
				case DataState::Host:
				case DataState::HostAndDevice:
					return m_HostData.m_Width;
				#ifdef CUDA_Support
				case DataState::Device:
					return m_DeviceData.m_Width;
				#endif
				default: return 0;
				}
			}

			int Height() const {
				switch (m_DataState) {
				case DataState::Host:
				case DataState::HostAndDevice:
					return m_HostData.m_Height;
				#ifdef CUDA_Support
				case DataState::Device:
					return m_DeviceData.m_Height;
				#endif
				default: return 0;
				}
			}

			const GPUStream& Stream() const {
				return m_Stream;
			}

			bool IsReadableOnHost() const {
				return m_DataState == DataState::Host || m_DataState == DataState::HostAndDevice;
			}

			bool IsReadableOnDevice() const {
				return m_DataState == DataState::Device || m_DataState == DataState::HostAndDevice;
			}

			bool IsWritableOnHost() const {
				return m_DataState == DataState::Host;
			}

			bool IsWritableOnDevice() const {
				return m_DataState == DataState::Device;
			}

			Rectangle<int> Bounds() const { return Rectangle<int>(0, 0, Width(), Height()); }

			HostFlags GetHostFlags() const { return m_HostData.GetFlags(); }			

			#pragma endregion

			#pragma region "CUDA Kernel Support"

			#ifdef CUDA_Support
			/// <summary>
			/// Call AfterKernelLaunch() to check for any errors in launching the kernel.
			/// This will check for all errors since the StartAsync() call.
			/// </summary>
			void	AfterKernelLaunchAux(const char* pszSourceFile, int nLine)
			{
				// Check for launch errors.  This works best if cudaGetLastError()
				// cleared the host thread's last error variable just before launching
				// the kernel.  This is accomplished by the StartAsync() call.
				::wb::cuda::Throwable(cudaPeekAtLastError(), pszSourceFile, nLine);
			}
			#endif

			#pragma endregion			

			#pragma region "Pixel Access"

			PixelType* GetHostScanlinePtr(int yy) { Synchronize(); return (PixelType*)(((byte*)m_HostData.m_pData) + yy * m_HostData.m_Stride); }
			const PixelType* GetHostScanlinePtr(int yy) const { Synchronize(); return (const PixelType*)(((byte*)m_HostData.m_pData) + yy * m_HostData.m_Stride); }
			
			int GetHostDataStride() const { return m_HostData.m_Stride; }
			int GetDeviceDataStride() const { return m_DeviceData.m_Stride; }

			PixelType& operator() (int xx, int yy) { ToHost(); Synchronize(); return *(PixelType*)(((byte*)m_HostData.m_pData) + xx * sizeof(PixelType) + yy * m_HostData.m_Stride); }
			const PixelType& operator() (int xx, int yy) const {
				if (!IsReadableOnHost()) throw NotSupportedException("Image must be readable on the host before calling const operator().");
				return *(PixelType*)(((byte*)m_HostData.m_pData) + xx * sizeof(PixelType) + yy * m_HostData.m_Stride);
			}

			#ifdef CUDA_Support
			PixelType* GetDeviceDataPtr() {
				if (m_DataState != DataState::Device && m_DataState != DataState::HostAndDevice)
					throw NotSupportedException("Image data has not been transferred to GPU before trying to access device pointer.");
				return m_DeviceData.m_pData;
			}

			const PixelType* GetDeviceDataPtr() const {
				if (m_DataState != DataState::Device && m_DataState != DataState::HostAndDevice)
					throw NotSupportedException("Image data has not been transferred to GPU before trying to access device pointer.");
				return m_DeviceData.m_pData;
			}
			#endif

			#if 0
			double GetPixelInterpolated(double X, double Y) const
			{
				typedef double PrecisionType;
				int x1 = (int)X, x2 = x1 + 1;
				int y1 = (int)Y, y2 = y1 + 1;
				PrecisionType XAlpha = std::fmod(X, 1.0);
				PrecisionType YAlpha = std::fmod(Y, 1.0);
				PrecisionType Sum = 0.0; PrecisionType Weight = 0.0, TotalWeight = 0.0;
				if (x1 >= 0 && y1 >= 0 && x1 < m_Width && y1 < m_Height) { Weight = (1.0 - XAlpha) * (1.0 - YAlpha);	Sum += operator()(x1, y1) * Weight; TotalWeight += Weight; }
				if (x2 >= 0 && y1 >= 0 && x2 < m_Width && y1 < m_Height) { Weight = XAlpha * (1.0 - YAlpha);			    Sum += operator()(x2, y1) * Weight; TotalWeight += Weight; }
				if (x1 >= 0 && y2 >= 0 && x1 < m_Width && y2 < m_Height) { Weight = (1.0 - XAlpha) * YAlpha;			    Sum += operator()(x1, y2) * Weight; TotalWeight += Weight; }
				if (x2 >= 0 && y2 >= 0 && x2 < m_Width && y2 < m_Height) { Weight = XAlpha * YAlpha;					        Sum += operator()(x2, y2) * Weight; TotalWeight += Weight; }
				if (TotalWeight < 1.0e-3) return 0.0;
				return Sum / TotalWeight;
			}
			#endif

			#pragma endregion

			#pragma region "Pixelwise Operations"

			void FillZero()
			{
				if (ModifyInHost())
				{
					memset(m_HostData.m_pData, 0, m_HostData.m_Stride * m_HostData.m_Height);
				}
				else
				{
					#ifdef CUDA_Support
					StartAsync();
					cudaThrowable(cudaMemsetAsync(m_DeviceData.m_pData, 0,
						m_DeviceData.m_Stride * m_DeviceData.m_Height, m_Stream));
					// AfterKernelLaunch(); unnecessarily, I believe, because cudaMemsetAsync() will check for errors.
					#else
					throw NotSupportedException();
					#endif
				}
			}

			/// <TODO>Optimization: this could be made less computationally intensive on the GPU: fill one line, then 
			/// switch to cudaMemsetAsync() to copy that line to all other lines of the image.  It becomes mostly DMA 
			/// operations, freeing the GPU to do other processing.</TODO>
			void Fill(PixelType value)
			{
				if (ModifyInHost())
				{
					for (int xx = 0; xx < m_HostData.m_Width; xx++) m_HostData.m_pData[xx] = value;
					for (int yy = 1; yy < m_HostData.m_Height; yy++)
					{
						memmove_s(GetHostScanlinePtr(yy), m_HostData.m_Stride, m_HostData.m_pData, m_HostData.m_Width * sizeof(PixelType));
					}
				}
				else
				{
					#ifdef CUDA_Support
					dim3 blocks, threads;
					GetSmallOpKernelParameters(blocks, threads);

					StartAsync();
					Launch_CUDAKernel(cuda::Kernel_Fill, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
						m_DeviceData.m_pData,
						cuda::img_format(m_DeviceData.m_Width, m_DeviceData.m_Height, m_DeviceData.m_Stride),
						value
					);
					AfterKernelLaunch();
					#else
					throw NotSupportedException();
					#endif
				}
			}

			#ifdef CUDA_Support
			#define InPlaceOperatorCodeTemplate_1Img1Scalar_Device(KernelName)	\
				{		\
					auto formatA = m_DeviceData.GetCudaFormat();			\
						\
					dim3 blocks, threads;									\
					GetSmallOpKernelParameters(blocks, threads);			\
						\
					StartAsync();											\
					Launch_CUDAKernel(KernelName, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream, \
						m_DeviceData.m_pData, formatA, \
						rhs													\
					);	\
					AfterKernelLaunch();									\
				}
			#else
			#define InPlaceOperatorCodeTemplate_1Img1Scalar_Device(KernelName) { throw NotSupportedException(); }
			#endif

			#define InPlaceOperatorCodeTemplate_1Img1Scalar(KernelName, HostOperation)		\
				{			\
					if (ModifyInHost())											\
					{		\
						for (int yy = 0; yy < m_HostData.m_Height; yy++)		\
						{	\
							PixelType* pA = GetHostScanlinePtr(yy);				\
							\
							for (int xx = 0; xx < m_HostData.m_Width; xx++, pA++) { HostOperation; }		\
						}	\
					}		\
					else InPlaceOperatorCodeTemplate_1Img1Scalar_Device(KernelName);						\
					return *this;												\
				}

			template<typename ScalarType> BaseImage& operator+=(ScalarType rhs) { InPlaceOperatorCodeTemplate_1Img1Scalar(cuda::Kernel_AddScalarInPlace, { *pA += rhs; }); }
			template<typename ScalarType> BaseImage& operator-=(ScalarType rhs) { InPlaceOperatorCodeTemplate_1Img1Scalar(cuda::Kernel_SubScalarInPlace, { *pA -= rhs; }); }
			template<typename ScalarType> BaseImage& operator*=(ScalarType rhs) { InPlaceOperatorCodeTemplate_1Img1Scalar(cuda::Kernel_MulScalarInPlace, { *pA *= rhs; }); }
			template<typename ScalarType> BaseImage& operator/=(ScalarType rhs) { InPlaceOperatorCodeTemplate_1Img1Scalar(cuda::Kernel_DivScalarInPlace, { *pA /= rhs; }); }
			#undef InPlaceOperatorCodeTemplate_1Img1Scalar

			#ifdef CUDA_Support
			#define InPlaceOperatorCodeTemplate_2Img_Device(KernelName)		\
				{															\
					auto formatA = m_DeviceData.GetCudaFormat();			\
					rhs.ToDevice();											\
					auto formatB = rhs.m_DeviceData.GetCudaFormat();		\
					if (formatA.size() != formatB.size()) throw FormatException("Image sizes must match for this operation.");			\
						\
					dim3 blocks, threads;									\
					GetSmallOpKernelParameters(blocks, threads);			\
						\
					rhs.StartAsync(m_Stream);								\
					StartAsync();											\
					Launch_CUDAKernel(KernelName, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,	\
						m_DeviceData.m_pData, formatA,						\
						rhs.GetDeviceDataPtr(), formatB						\
					);	\
					AfterKernelLaunch();									\
				}
			#else
			#define InPlaceOperatorCodeTemplate_2Img_Device(KernelName) { throw NotSupportedException(); }
			#endif

			#define InPlaceOperatorCodeTemplate_2Img(KernelName, HostOperation)		\
				{			\
					if (ModifyInHost(rhs))										\
					{		\
						for (int yy = 0; yy < m_HostData.m_Height; yy++)		\
						{	\
							PixelType* pA = GetHostScanlinePtr(yy);				\
							PixelType* pB = (PixelType*)rhs.GetHostScanlinePtr(yy);	\
							\
							for (int xx = 0; xx < m_HostData.m_Width; xx++, pA++, pB++) { HostOperation; }		\
						}	\
					}		\
					else InPlaceOperatorCodeTemplate_2Img_Device(KernelName);	\
					return *this;												\
				}

			// These are distinguished from the template operators above with a ScalarType because of the by-reference argument.
			BaseImage& operator+=(FinalType& rhs) { InPlaceOperatorCodeTemplate_2Img(cuda::Kernel_AddInPlace, { *pA += *pB; }); }
			BaseImage& operator-=(FinalType& rhs) { InPlaceOperatorCodeTemplate_2Img(cuda::Kernel_SubInPlace, { *pA -= *pB; }); }
			BaseImage& operator*=(FinalType& rhs) { InPlaceOperatorCodeTemplate_2Img(cuda::Kernel_MulInPlace, { *pA *= *pB; }); }
			BaseImage& operator/=(FinalType& rhs) { InPlaceOperatorCodeTemplate_2Img(cuda::Kernel_DivInPlace, { *pA /= *pB; }); }
			#undef InPlaceOperatorCodeTemplate			

			#ifdef CUDA_Support
			#define NewOperatorCodeTemplate_2Img_Device(KernelName)		\
				{		\
					auto ret = FinalType::NewDeviceImage(m_DeviceData.m_Width, m_DeviceData.m_Height, Stream(), GetHostFlags());		\
					auto formatA = m_DeviceData.GetCudaFormat();			\
					rhs.ToDevice();											\
					auto formatB = rhs.m_DeviceData.GetCudaFormat();		\
					if (formatA.size() != formatB.size()) throw FormatException("Image sizes must match for this operation.");			\
					auto formatResult = ret.m_DeviceData.GetCudaFormat();	\
						\
					dim3 blocks, threads;									\
					GetSmallOpKernelParameters(blocks, threads);			\
						\
					rhs.StartAsync(m_Stream);								\
					StartAsync();											\
					Launch_CUDAKernel(KernelName, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,	\
						m_DeviceData.m_pData, formatA,						\
						rhs.GetDeviceDataPtr(), formatB,					\
						ret.GetDeviceDataPtr(), formatResult				\
					);	\
					AfterKernelLaunch();									\
					return ret;												\
				}
			#else
			#define NewOperatorCodeTemplate_2Img_Device(KernelName) { throw NotSupportedException(); }
			#endif

			#define NewOperatorCodeTemplate_2Img(KernelName, HostOperation)		\
				{			\
					if (WouldModifyInHost(rhs))									\
					{		\
						auto ret = FinalType::NewHostImage(m_HostData.m_Width, m_HostData.m_Height, Stream(), GetHostFlags());		\
						for (int yy = 0; yy < m_HostData.m_Height; yy++)		\
						{	\
							PixelType* pA = GetHostScanlinePtr(yy);				\
							PixelType* pB = (PixelType*)rhs.GetHostScanlinePtr(yy);	\
							PixelType* pResult = (PixelType*)ret.GetHostScanlinePtr(yy);	\
							\
							for (int xx = 0; xx < m_HostData.m_Width; xx++, pA++, pB++) { HostOperation; }		\
						}	\
						return ret;												\
					}		\
					else NewOperatorCodeTemplate_2Img_Device(KernelName);		\
				}

			// These are distinguished from the template operators above with a ScalarType because of the by-reference argument.
			FinalType operator+(FinalType& rhs) { NewOperatorCodeTemplate_2Img(cuda::Kernel_Add, { *pResult = *pA + *pB; }); }
			FinalType operator-(FinalType& rhs) { NewOperatorCodeTemplate_2Img(cuda::Kernel_Sub, { *pResult = *pA - *pB; }); }
			FinalType operator*(FinalType& rhs) { NewOperatorCodeTemplate_2Img(cuda::Kernel_Mul, { *pResult = *pA * *pB; }); }
			FinalType operator/(FinalType& rhs) { NewOperatorCodeTemplate_2Img(cuda::Kernel_Div, { *pResult = *pA / *pB; }); }
			#undef NewOperatorCodeTemplate_2Img

			#ifdef CUDA_Support
			#define InPlaceUnaryOperatorCodeTemplate_Device(KernelName)		\
				{		\
					auto formatA = m_DeviceData.GetCudaFormat();			\
						\
					dim3 blocks, threads;									\
					GetSmallOpKernelParameters(blocks, threads);			\
						\
					StartAsync();											\
					Launch_CUDAKernel(KernelName, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,	\
						m_DeviceData.m_pData, formatA						\
					);	\
					AfterKernelLaunch();									\
				}
			#else
			#define InPlaceUnaryOperatorCodeTemplate_Device(KernelName) { throw NotSupportedException(); }
			#endif

			#define InPlaceUnaryOperatorCodeTemplate(KernelName, HostOperation)	\
				{			\
					if (ModifyInHost())											\
					{		\
						for (int yy = 0; yy < m_HostData.m_Height; yy++)		\
						{	\
							PixelType* pA = GetHostScanlinePtr(yy);				\
							\
							for (int xx = 0; xx < m_HostData.m_Width; xx++, pA++) { HostOperation; }		\
						}	\
					}		\
					else InPlaceUnaryOperatorCodeTemplate_Device(KernelName);	\
					return *this;												\
				}

			#undef InPlaceUnaryOperatorCodeTemplate

			FinalType& SaturateInPlace(PixelType MinValue, PixelType MaxValue)
			{
				if (ModifyInHost())
				{
					for (int yy = 0; yy < m_HostData.m_Height; yy++)
					{
						PixelType* pA = GetHostScanlinePtr(yy);

						for (int xx = 0; xx < m_HostData.m_Width; xx++, pA++) {
							*pA = (*pA < MinValue) ? MinValue : ((*pA > MaxValue) ? MaxValue : *pA);
						}
					}
				}
				else
				{
					#ifdef CUDA_Support
					auto formatA = m_DeviceData.GetCudaFormat();

					dim3 blocks, threads;
					GetSmallOpKernelParameters(blocks, threads);

					StartAsync();
					Launch_CUDAKernel(cuda::Kernel_SaturateInPlace, blocks, threads, /*dynamic shared memory=*/ 0, m_Stream,
						m_DeviceData.m_pData, formatA, MinValue, MaxValue
					);
					AfterKernelLaunch();
					#else		
					throw NotSupportedException();
					#endif
				}
				return (FinalType&)*this;
			}

			#pragma endregion	// Pixelwise Operations

			#pragma region "Convolution Operations"			

			#ifdef NPP_Support
			template<typename KernelType> FinalType& Convolve(FinalType& Filtered, const ConvolutionKernel<KernelType>& Kernel, const Rectangle<int>& ROI);

			/// <summary>
			/// Applies a general convolution kernel to an image and returns a new image.
			/// Note: Not all KernelTypes are supported.  Int32 and float are generally supported,
			/// and double is supported for double images.
			/// </summary>			
			/// <param name="Kernel">The general convolution kernel to apply.</param>			
			/// <returns></returns>
			template<typename KernelType> FinalType Convolve(const ConvolutionKernel<KernelType>& Kernel, const Rectangle<int>& ROI);

			template<typename KernelType> FinalType& Convolve(FinalType& Filtered, const ConvolutionKernel<KernelType>& Kernel) { return Convolve(Filtered, Kernel, Rectangle<int>::Whole()); }

			template<typename KernelType> FinalType Convolve(const ConvolutionKernel<KernelType>& Kernel) { return Convolve(Kernel, Rectangle<int>::Whole()); }			

			#endif

			#pragma endregion

			#pragma region "Transforms"

			FinalType& FlipVerticallyTo(FinalType& dst);			
			FinalType FlipVerticallyTo(HostFlags Flags = HostFlags::Retain);
			
			#ifdef NPP_Support			

			FinalType& ResizeTo(FinalType& dst, 
				const Rectangle<int>& ToROI = Rectangle<int>::Whole(), const Rectangle<int>& FromROI = Rectangle<int>::Whole(), 
				InterpolationMethods Method = InterpolationMethods::Linear);			
			FinalType ResizeTo(const Rectangle<int>& ToROI, const Rectangle<int>& FromROI = Rectangle<int>::Whole(), InterpolationMethods Method = InterpolationMethods::Linear,
				HostFlags Flags = HostFlags::Retain);

			#endif

			/// <summary>
			/// Copies a ROI from one image into another image.  Can also copy a ROI from within one image to a new location within
			/// the same image, but the ROIs may not overlap.
			/// </summary>
			/// <param name="dst">Destination image.  Must be large enough to accomodate the ROI plus ToX and ToY offset.</param>
			/// <param name="ToX">X ordinate in the destination image to copy the top-leftmost pixel of the ROI into.</param>
			/// <param name="ToY">Y ordinate in the destination image to copy the top-leftmost pixel of the ROI into.</param>
			/// <param name="FromROI">The ROI in this (source) image to copy from.  If anything except Rectangle&lt;int&gt;::Whole(), 
			/// this defines a cropping operation.</param>
			/// <returns>Reference to dst image.</returns>
			FinalType& CopyTo(FinalType& dst, Rectangle<int> FromROI = Rectangle<int>::Whole(), int ToX = 0, int ToY = 0);			

			/// <summary>
			/// Crops this image from the specified coordinates (all inclusive) using the current 
			/// stream.
			/// </summary>
			/// <param name="x1">Leftmost ordinate to include in the cropped image.</param>
			/// <param name="y1">Topmost ordinate to include in the cropped image.</param>
			/// <param name="Width">Width to crop to, in pixels.</param>
			/// <param name="Height">Height to crop to, in pixels.</param>
			/// <param name="Flags">Flags to apply to the returned image.  By default, the same flags
			/// as the source image are used.</param>
			/// <returns>An image that is a cropped version of the source image.</returns>
			FinalType CropTo(int x1, int y1, int Width, int Height, HostFlags Flags = HostFlags::Retain);			

			/// <summary>
			/// Crops this image from the specified coordinates using the current 
			/// stream.
			/// </summary>			
			/// <param name="FromROI">The ROI in this (source) image to copy from.</param>
			/// <param name="Flags">Flags to apply to the returned image.  By default, the same flags
			/// as the source image are used.</param>
			/// <returns>An image that is a cropped version of the source image.</returns>
			FinalType CropTo(const Rectangle<int>& FromROI, HostFlags Flags = HostFlags::Retain);

			#pragma endregion			
		};

		#pragma endregion		// "Image Foundations"		

		#pragma region "Generalized real and complex templates"

		/// <summary>
		/// RealImage (real, single-channel image) is an intermediate class.  Descending from BaseImage,
		/// it provides functionality that is specific to real-valued single-channel images only.  Examples
		/// of classes that descend from RealImage are Image&lt;byte&gt;, Image&lt;int&gt;, 
		/// Image&lt;float&gt;, etc.
		/// </summary>		
		template<typename PixelType, typename FinalType> class RealImage : public BaseImage<PixelType, FinalType>
		{
			typedef BaseImage<PixelType, FinalType> base;

		public:
			// The MSVC compiler will not look in dependent base classes for
			// nondependent names.  These using declarations will resolve the issue.			
			using base::m_DataState;
			using base::m_HostData;
			#ifdef CUDA_Support
			using base::m_DeviceData;
			#endif
		
		protected:
			RealImage(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			RealImage() = default;
			RealImage(RealImage&&) = default;
			RealImage(RealImage&) = delete;
			RealImage& operator=(RealImage&&) = default;
			RealImage& operator=(RealImage&) = delete;

			/// <summary>
			/// Calculates absolute value elementwise.  
			/// </summary>
			/// <param name="dst">The destination image to place the result in.</param>
			/// <returns>Reference to the destination image.</returns>
			RealImage& AbsoluteTo(RealImage& dst);

			/// <summary>
			/// Calculates absolute value elementwise and returns a new image with the result.
			/// </summary>
			/// <returns>A new image containing the absolute value.</returns>
			FinalType Absolute(HostFlags Flags = HostFlags::Retain);

			/// <summary>
			/// Calculates the absolute value elementwise and replaces this image with the absolute
			/// value at each pixel.
			/// </summary>
			/// <returns>Reference to this image.</returns>
			FinalType& AbsoluteInPlace();

			/// <summary>
			/// Calculates the sum of the pixels of the image, using the specified type for the accumulator
			/// and final result.
			/// </summary>
			template<typename SumType> SumType Sum(Rectangle<int> ROI = Rectangle<int>::Whole());

			/// <summary>
			/// Calculates the arithmetic mean of the image, using the specified type for the accumulator
			/// and final result.
			/// </summary>
			template<typename MeanType> MeanType Mean(Rectangle<int> ROI = Rectangle<int>::Whole());

			/// <summary>
			/// Calculates the arithmetic mean and standard deviation of the image.
			/// </summary>
			void MeanAndStdDev(double& Mean, double& StdDev);

			/// <summary>
			/// Calculates the max value of pixels of the image.
			/// </summary>
			PixelType Max();

			/// <summary>
			/// Calculates the min value of pixels of the image.
			/// </summary>
			PixelType Min();

			FinalType& FilterSobelHorizontalTo(FinalType& Filtered);
			FinalType& FilterSobelVerticalTo(FinalType& Filtered);
		};

		/// <summary>
		/// ComplexImage (complex, single-channel image) is an intermediate class.  Descending from BaseImage,
		/// it provides functionality that is specific to complex-valued single-channel images only.  Examples
		/// of classes that descend from ComplexImage are Image&lt;thrust::complex&lt;float&gt;&gt; and 
		/// Image&lt;thrust::complex&lt;double&gt;&gt;.
		/// </summary>		
		template<typename PixelType, typename RealPixelType, typename FinalType, typename RealFinalType> class ComplexImage : public BaseImage<PixelType, FinalType>
		{
			typedef BaseImage<PixelType, FinalType> base;

		public:
			// The MSVC compiler will not look in dependent base classes for
			// nondependent names.  These using declarations will resolve the issue.			
			using base::m_DataState;
			using base::m_HostData;
			#ifdef CUDA_Support
			using base::m_DeviceData;
			#endif
		
		protected:
			ComplexImage(GPUStream Stream, HostFlags HostFlags) : base(Stream, HostFlags) { }
		public:
			ComplexImage() = default;
			ComplexImage(ComplexImage&&) = default;
			ComplexImage(ComplexImage&) = delete;
			ComplexImage& operator=(ComplexImage&&) = default;
			ComplexImage& operator=(ComplexImage&) = delete;
			
			//ComplexImage(ComplexImage&& mv) noexcept : base(std::move(mv)) { }

			/// <summary>
			/// SetReal() updates the real component of every pixel with the value from src while retaining the imaginary component.
			/// </summary>			
			FinalType& SetReal(RealImage<RealPixelType, RealFinalType>& src);

			/// <summary>
			/// SetImag() updates the imaginary component of every pixel with the value from src while retaining the real component.
			/// </summary>
			FinalType& SetImag(RealImage<RealPixelType, RealFinalType>& src);
			
			RealImage<RealPixelType, RealFinalType>& GetRealTo(RealImage<RealPixelType, RealFinalType>& dst);
			RealFinalType GetReal(HostFlags Flags = HostFlags::Retain);
			RealImage<RealPixelType, RealFinalType>& GetImagTo(RealImage<RealPixelType, RealFinalType>& dst);
			RealFinalType GetImag(HostFlags Flags = HostFlags::Retain);

			RealImage<RealPixelType, RealFinalType>& AbsoluteTo(RealImage<RealPixelType, RealFinalType>& dst);
			RealFinalType Absolute(HostFlags Flags = HostFlags::Retain);

			/// <summary>
			/// Calculates angle of the complex value, elementwise.  
			/// </summary>
			/// <param name="dst">The destination image to place the result in.</param>
			/// <returns>Reference to the destination image.</returns>
			RealImage<RealPixelType, RealFinalType>& AngleTo(RealImage<RealPixelType, RealFinalType>& dst);

			/// <summary>
			/// Calculates angle of the complex value, elementwise and returns a new image with the result.
			/// </summary>
			/// <returns>A new image containing the absolute value.</returns>
			RealFinalType Angle(HostFlags Flags = HostFlags::Retain);

			ComplexImage& ConjugateTo(ComplexImage& dst);
			FinalType Conjugate(HostFlags Flags = HostFlags::Retain);
			FinalType& ConjugateInPlace();

			/// <summary>
			/// Calculates the sum of the pixels of the image, using the specified type for the accumulator
			/// and final result.
			/// </summary>
			//template<typename SumType> SumType Sum();		// Temporarily disabled due to a compiler issue.  See function.

			/// <summary>
			/// Calculates the arithmetic mean of the image, using the specified type for the accumulator
			/// and final result.
			/// </summary>
			//template<typename MeanType> MeanType Mean();	// Temporarily disabled because Sum() is temporarily disabled.
		};

		#pragma endregion

		#pragma region "Troubleshooting & Tools"

		template<typename PixelType> inline std::string display_string(PixelType val) { return std::to_string(val); }
		template<> inline std::string display_string(byte val) { return std::to_string((int)val); }
		template<> inline std::string display_string(thrust::complex<float> val) {
			if (val.imag() == 0.0f) return std::to_string(val.real());
			return std::to_string(val.real()) + ((val.imag() >= 0.0f) ? "+i" : "-i") + std::to_string(abs(val.imag())); 
		}
		template<> inline std::string display_string(thrust::complex<double> val) {
			if (val.imag() == 0.0f) return std::to_string(val.real());
			return std::to_string(val.real()) + ((val.imag() >= 0.0f) ? "+i" : "-i") + std::to_string(abs(val.imag()));
		}
		
		template<typename OrdinateType> inline std::string to_string(const Rectangle<OrdinateType>& rect)
		{
			std::stringstream out;
			out.precision(3);
			out << "rectangle(x=" << rect.X << ", y=" << rect.Y << ", width=" << rect.Width << ", height=" << rect.Height << ")";
			return out.str();
		}

		template<typename OrdinateType> inline std::ostream& operator<<(std::ostream& os, const Rectangle<OrdinateType>& rect)
		{
			os << to_string(rect);
			return os;
		}

		template<typename PixelType, typename FinalType> inline std::string to_string(const BaseImage<PixelType, FinalType>& src)
		{
			std::stringstream out;
			out.precision(3);
			out << "image(width=" << src.Width() << ", height=" << src.Height() << "): \n";

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
			return out.str();
		}

		template<typename PixelType, typename FinalType> inline std::ostream& operator<<(std::ostream& os, const BaseImage<PixelType, FinalType>& img)
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

#include "Images_Convolution.h"
#include "Images_Transforms.h"

#endif	// __WBImages_Base_h__

//	End of Images_Base.h

