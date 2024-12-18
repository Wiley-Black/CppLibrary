/////////
//	ndarray_view.h
/////////
/*	A view provides an alternative but compatible window into the same data array but with
*	some characteristics altered such as accessing only a subset, skipping elements, or
*	casting-in-place to a new type.
* 
*	Row/Column-Major Problem Statement _____
*
*	By default, this library uses row-major ordering.  Column-major order can be supported through
*	atypical strides.  For example, a row-major byte-element matrix laid out with strides { 6, 1 } 
*	could be represented as { 1, 6 } in column-major format.  This would indicate that incrementing 
*	in the last dimension ("x") adds 6 bytes to the pointer while incrementing in the higher 
*	dimension ("y") adds 1 byte to the pointer.
*
*	One place where this has utility is in a transpose() function implementation.  Such an
*	implementation could directly be a view that alters the strides utilized into column-major
*	format upon the same underlying data.
*
*	a =	  {	{ 1,	2,	3 },
*			{ 4,	5,	6 } }
*	a.strides = { 3, 1 }
*
*	a.transpose() :=  {	{ 1, 4 },
*						{ 2, 5 },
*						{ 3, 6 } }
*	a.transpose().strides = { 1, 3 } if represented as a view into a.
*
*	Row/Column-Major Implementation _____
*
*	A key rule must be observed: the offset of any element in bytes from the start of the data
*	buffer can be calculated as the dot product between the strides and the element position.
*	For example, in the above transpose example, the position of element '3' in matrix a is
*	{ 0, 2 }.  The dot product of this position with the strides yields 3*0 + 2*1 = 2.  For
*	the transpose() view into the same matrix, the dot product of position { 2, 0 } with
*	the transposed strides { 1, 3 } yields 2*1 + 0*3 = 2 again.
*/

#ifndef __WB_ndarray_view_h__
#define __WB_ndarray_view_h__

/** Dependencies **/

#include "../../wbFoundation.h"
#include "ndarray_memory.h"

namespace wb { namespace math { namespace memory {	

	#pragma region "NDArrayBufferView class"

	class NDArrayBufferView
	{		
	public:
		/// <summary>
		/// strides specifies the number of bytes between elements of each dimension.  The strides 
		/// may or may not lead to a contiguous layout ("tightly packed") in any dimension.  The 
		/// last dimension will typically be equal to the size of an individual element, and 
		/// higher-level structures such as NDArray may require this.  It is not unusual for
		/// higher-level structures to require dimensions other than the last to be padded to a
		/// certain bit-length for computational optimization and this can lead to a non-contiguous
		/// layout.
		/// 
		/// For example, in the example of a 3x64x64 matrix representing a small RGB image, the 
		/// strides might be {64*64*4, 64*4, 4} for the case of a float element type.  This 
		/// would specify that the last dimension (the 3rd) contains only 4 bytes between 
		/// elements, as is true for a 'float' element type.  The next dimension (2nd) would 
		/// contain 64 elements and therefore has 64*4 bytes between each iteration of that 
		/// dimension.
		/// 
		/// In another example, assume that a 4x4 matrix has 16-bit element types (Int16) but
		/// carries an alignment requirement such that matrix rows be aligned to a 128-bit boundary.
		/// The shape in that example would be (4, 4) but strides would be (16, 2).
		/// 
		/// The dot product of a requested element position and the strides vector should always
		/// produces the number of bytes relative to the start of the data buffer (plus offset)
		/// necessary to access the element at the requested position:
		/// 
		///		element_offset_in_bytes = p_buffer->p_data + offset + dot_product(strides, position)
		/// </summary>
		vector<size_t>	strides;

		/// <summary>
		/// Provides the number of bytes of offset from the start of the buffer's memory to the first element's
		/// start position.  The offset can be used in combination with reducing shape to provide a window
		/// into a buffer that does not start with the same element as the buffer starts with.
		/// </summary>
		size_t	offset;

		/// <summary>
		/// A pointer to the buffer underlying this view.
		/// </summary>
		shared_ptr<NDArrayBuffer>	p_buffer;

		NDArrayBufferView(shared_ptr<NDArrayBuffer> p_buffer_) : p_buffer(p_buffer_)
		{
			offset = 0;
		}

		bool is_outdated() const { return p_buffer->outdated; }
		bool mark_outdated() const { return p_buffer->outdated = true; }
		bool mark_updated() const { return p_buffer->outdated = false; }

		bool is_host_memory() const { return p_buffer->is_host_memory(); }
		bool is_device_memory() const { return p_buffer->is_device_memory(); }		

		bool is_pending_async_read() const { return p_buffer->is_pending_async_read(); }
		bool is_pending_async_write() const { return p_buffer->is_pending_async_write(); }
		void add_pending_async_read(cuda::GPUStream stream) { p_buffer->add_pending_async_read(stream); }
		void add_pending_async_write(cuda::GPUStream stream) { p_buffer->add_pending_async_write(stream); }
		void finish_reads() { p_buffer->finish_reads(); }
		void finish_writes() { p_buffer->finish_writes(); }

		bool is_allocated() const { return p_buffer->is_allocated(); }
		bool can_allocate() const { return p_buffer->can_allocate(); }
		bool can_write() const { return p_buffer->can_write(); }

		void allocate(const ndarray_allocation_t& requirements)
		{
			// The strides aren't stored at the buffer-level and strides can vary 
			// between different views into the same buffer.  However, the buffer 
			// may have alignment requirements, sometimes dynamic, and thus it is
			// necessary to ask the buffer to accomplish calculation of strides.			
			strides = p_buffer->calculate_strides(requirements);
			p_buffer->allocate(requirements, strides);
		}

		/// <summary>
		/// Retrieves a pointer to the data for this buffer.  The pointer can be either in host
		/// or device memory depending on the buffer.
		/// </summary>		
		void* get_pointer()
		{
			if (is_host_memory()) return dynamic_cast<HostBuffer*>(p_buffer.get())->get_host_pointer();
			#ifdef CUDA_Support
			else if (is_device_memory()) return dynamic_cast<CUDADeviceBuffer*>(p_buffer.get())->get_device_pointer();
			#endif
			return nullptr;			
		}

		/// <summary>
		/// Retrieves the next_stream for this buffer if there is one.  For a host buffer,
		/// GPUStream::None() is returned.
		/// </summary>		
		cuda::GPUStream get_next_stream()
		{
			#ifdef CUDA_Support
			if (is_device_memory()) return dynamic_cast<CUDADeviceBuffer*>(p_buffer.get())->next_stream;
			#endif
			return cuda::GPUStream::None();			
		}

		
	};
	
	#pragma endregion

} } }		// end namespaces

#endif	// __WB_ndarray_view_h__

//	End of ndarray_view.h
