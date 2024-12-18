/////////
//	ndarray_definitions.h
/////////
/**	API ______
*
*	A "challenge" example that covers most everything conceptually:
*		auto my_stream = GPUStream();
*		auto my_array = ndarray<float>({3, 64, 64}, {
*			ReadableExistingHostBuffer(memory_buffer_ro, {96*64, 4}),
*			WritableExistingHostBuffer(memory_buffer_rw, {64*64, 4}, false),
*			my_stream,
*			host_stream
*		});
*
*	The my_array example array is using a read-only existing host buffer, an existing writable host buffer, a specific
*	GPU stream, and allows allocation of new host memory if needed.  The readable, existing host buffer is assumed to
*	provide the initial data for this array because an "existing buffer" is presumed to have existing data by default.
*
*	The third argument to the WritableExistingHostBuffer() is the "initial_data" flag and setting it false indicates that,
*	while the buffer is pre-existing and writable, it does not contain initial data.  Had both the ExistingHostBuffer()'s
*	in this example used the default "initial_data = true" flag, then it would be assumed that both contain the same, up-to-date
*	data and that data could be read from either existing buffer.  For an "in-place" operation the WritableExistingHostBuffer
*	would be utilized if both initial_data were true.  However with intial_data of false for the writable buffer, an "in-place"
*	operation is not allowed and the data would first have to be copied to one of the writable buffers.
*
*	The inclusion of host in the list would seem unnecessary given the WritableExistingHostBuffer() being provided, however it
*	does carry meaning.  The one operation not permissible on WritableExistingHostBuffer() is to allocate new memory.  If an
*	operation required resizing the array, then new memory would be needed.  Without host this would cause an exception.  In
*	most cases, it makes sense to to the resize operation "out-of-place" and just start a new ndarray.  A corner case where
*	this might be useful would be if you didn't know the needs up-front.  For instance, perhaps sometimes the result comes
*	back as smaller and can fit into the existing host buffer but in other instances it doesn't.  Optimal, in such a case,
*	might be to permit reallocation of memory _if needed_ by providing host as the final stream.  Since it comes last in the
*	list, it is considered last as an option.
*
*	Implementation ______
*
*	The ndarray constructor (and other buffer-makers such as the zeros function) must accept an argument taking in
*	some common type or a vector of that common type.  Some of the cases it needs to accept include:
*		host_stream			(Constant)
*		host_pinned			(Constant)
*		any_gpu_stream		(Constant)
*		ReadOnlyExistingHostBuffer(void*, vector<size_t>&, bool initial_data = true)
*		WritableExistingHostBuffer(void*, vector<size_t>&, bool initial_data = true)
*		ReadOnlyExistingDeviceBuffer(void*, vector<size_t>&, bool initial_data = true)
*		WritableExistingDeviceBuffer(void*, vector<size_t>&, bool initial_data = true)
*		GPUStream object
*		custom buffer-creation implementations
*
*	In order to accept this variety of arguments, an interface is provided, INDArrayAllocator.  The constants
*	can be anything that implements this interface.  The Existing...Buffer() arguments could be a function
*	that returns an object that implements INDArrayAllocator, but such a function is by definition a constructor.
*	The GPUStream object can implement the INDArrayAllocator.
*
*	In all cases, the data describing the buffer is not passed into the ndarray constructor through this argument
*	or vector but factories are passed in instead.  The actual act of creating the buffer can be initiated by
*	the ndarray constructor, which is helpful because the ndarray constructor has additional information available
*	to provide to the factory such as shape.
*
*	The ndarray constructor should not be responsible for freeing the memory of any of the stream-related arguments
*	passed in.  For example, ReadOnlyExistingHostBuffer() should not provide a pointer to be dealt with.  Instead,
*	these objects can be passed into the ndarray constructor, utilized by it, and the ndarray constructor can then
*	become responsible for the NDArrayBuffer-derived object that the factor produces.
*
*	The NDArrayBuffer class cannot be a template class.  That's because the new_ndarray_buffer() function is
*	virtual, and it wouldn't make sense to have a function (as opposed to a class) be templated but also virtual.
*	This also prevents us from passing in the ndarray itself to the new_ndarray_buffer() function.  The
*	ndarray_allocation_t structure provides any information that might be needed as an alternative.
*
*	Implementation Layout ______
*
*	ndarray
*		vector<size_t>				shape;
*		vector<NDArrayBuffer*>		p_buffers;
*
*	NDArrayBuffer
*		vector<size_t>				strides;
*
*	ndarray_allocation_t
*		vector<size_t>&				shape;
*		vector<size_t>&				strides;
*
*	INDArrayAllocator
*		virtual NDArrayBuffer*		new_ndarray_buffer() = 0;
*
*	GPUStream : INDArrayAllocator
*
*	GenericNDArrayAllocator<typename BufferType> : INDArrayAllocator
*		NDArrayBuffer*				new_ndarray_buffer() override {
*			return new BufferType();
*		}
*
*	HostBuffer : NDArrayBuffer
*		void* get_host_pointer()
*	static GenericNDArrayAllocator<HostBuffer> host_stream;		// When CUDA_Support is absent.
*
*	When CUDA_Support is present:
*	CUDAHostBuffer : HostBuffer
*		flags [template argument]
*	static GenericNDArrayAllocator<CUDAHostBuffer<HostFlags::line_aligned>> host_stream;
*	static GenericNDArrayAllocator<CUDAHostBuffer<HostFlags::pinned>> host_pinned;
*	static GenericNDArrayAllocator<CUDAHostBuffer<HostFlags::pinned | HostFlags::portable>> host_portable;
*
*	CUDADeviceBuffer : ndarray_buffer_t
*		void* get_device_pointer()
*/

#ifndef __WB_ndarray_definitions_h__
#define __WB_ndarray_definitions_h__

/** Dependencies **/

#include "../../wbFoundation.h"
// Note: cannot depend on GPU.h because GPUStream descends from INDArrayAllocator, defined here.

#ifdef FreeImage_Support
#pragma comment(lib, "FreeImage.lib")
#include "FreeImage.h"
#endif

namespace wb { namespace math { namespace memory {

#pragma region "Definitions"

	// Design Notes: HostFlags properly belongs in the wb::images::memory::HostImageData class, but this is too 
	// cumbersome as it gets used often.  Also, embedding it in a class requires that it not be a class enum,
	// as AddFlagSupport() won't work in an embedded setup like that.  Also when I tried it inside the HostImageData
	// class, I had to reference it as:
	//			typedef typename memory::HostImageData<PixelType>::Flags HostFlags;			// Yucky, C++.
	// And one last problem is occasions where the template parameter might not really need to be specified just to
	// give the flags.

	/// <summary>
	/// HostFlags specify rules regarding memory allocation for the host image data.  HostFlags
	/// can be combined together (|).  Pinned must be specified for Portable, Mapped, or WriteCombined.
	/// </summary>		
	enum class HostFlags
	{
		none,				// Allocates ordinary memory.  Not recommended for transfer to device RAM.		
		pinned,				// Allocates pinned memory, accessable directly by GPU device.
		portable,			// Considered pinned for all CUDA contexts, not just the one that performed the allocation.
		mapped,				// Maps the allocation into the CUDA address space.
		write_combined,		// Allocates write-combined (WC), which can be transferred across the PCI Express bus more quickly
		// on some systems, but cannot be read efficiently by most CPUs.  Good for host->device
		// transfers.
	};
	AddFlagSupport(HostFlags);		// Enables the use of |, &, ^, |=, &=, and ^= operators on this enumeration.

	enum class ElementTypeID
	{
		Other,						// Indicates a type that is not "built-in".
		Int8,
		Int16,
		Int32,
		Int64,
		UInt8,
		UInt16,
		UInt32,
		UInt64,
		Float32,
		Float64,
		Complex32x2,
		Complex64x2,
		RGB24,
		RGBA
	};

	inline int bits_per_element_from_ID(ElementTypeID id)
	{
		switch (id)
		{
		case ElementTypeID::Int8: return 8;
		case ElementTypeID::Int16: return 16;
		case ElementTypeID::Int32: return 32;
		case ElementTypeID::Int64: return 64;
		case ElementTypeID::UInt8: return 8;
		case ElementTypeID::UInt16: return 16;
		case ElementTypeID::UInt32: return 32;
		case ElementTypeID::UInt64: return 64;
		case ElementTypeID::Float32: return 32;
		case ElementTypeID::Float64: return 64;
		case ElementTypeID::Complex32x2: return 64;
		case ElementTypeID::Complex64x2: return 128;
		case ElementTypeID::RGB24: return 24;
		case ElementTypeID::RGBA: return 32;
		default: throw NotSupportedException("The bits-per-pixel for the requeste element type is not available.");
		}
	}

#if 0
	/// <summary>
	/// The DataResponsibility type is used in accounting for who has responsibility for 
	/// freeing image memory, and for what access we are allowed on that memory.  Any memory
	/// allocated by the ImageData classes is going to RdWrR responsibility.  Image data 
	/// that was given via pointer from outside the class may be blocked from certain
	/// operations and would require a copy into a new image before manipulation.
	/// </summary>
	enum class DataResponsibility
	{
		None,				// No data yet, no responsibilities.
		RdWrR,				// Read and write are allowed.  Responsible for memory.
		RdWr,				// Read and write are allowed.  Cannot resize or reallocate memory.
		Rd					// Read-only.
	};
#endif

#pragma endregion

#pragma region "Requirements Specifier"

	struct ndarray_allocation_t
	{
		/// <summary>
		/// shape specifies the logical shape of the NDArray memory in number of elements.  For 
		/// example, a 64x64 RGB image might be specified with shape (3, 64, 64).  The first 
		/// dimension (in this example) is used for channel/color, the second is the height, 
		/// and the third is the width.  The shape is indifferent to the elements chosen, such 
		/// that the same shape would apply whether the underlying element is 'byte' or 'float' 
		/// and is irrespective of the underlying memory layout.
		/// </summary>
		vector<size_t>	shape;

		/// <summary>
		/// Indicates the size, in bytes, of a single element in the ndarray to be allocated.
		/// For example, if allocating an ndarray to contain UInt32 elements, this value should
		/// be sizeof(UInt32) which is 4 bytes.
		/// </summary>
		size_t	element_size;

		/// <summary>
		/// If using a standard element type (i.e. UInt16, Int64, or RGB24) it can be specified
		/// in element_type and some libraries need this to be able to operate on the buffer.
		/// If it is not a standard element type, use ElementTypeID::Other instead.
		/// </summary>
		ElementTypeID element_type;
	};

#pragma endregion

#pragma region "Allocation Interfaces"

	// Forward declaration
	class NDArrayBuffer;

	class INDArrayAllocator
	{
	public:
		virtual NDArrayBuffer* new_ndarray_buffer() = 0;
	};

	template<typename BufferType, int row_alignment = 16 /*bytes*/>
	class GenericNDArrayAllocator : public INDArrayAllocator
	{
	public:
		NDArrayBuffer* new_ndarray_buffer() override {
			return new BufferType(std::vector<size_t>({ row_alignment }));
		}
	};

#pragma endregion	

} } }	// End namespaces

#endif	// __WB_ndarray_definitions_h__

//	End of ndarray_definitions.h
