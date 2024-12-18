/////////
//	ndarray_memory.h
/////////
//	See ndarray_definitions.h for much of the thoughts behind NDArrayBuffer.
//
/*	Asynchronous Problem Statement _____
*
*	Consider asynchronous operations happening between two ndarray objects or between buffers within
*	one ndarray object.
*
*		auto a = ndarray<float>::zeros({32, 32}, host_pinned);
*		auto b = a;					// Host copy, potentially asynchronous, now initiated.
*		// other work could be completed while waiting here.
*		a(5, 5) = 9;				// Host-side write, must complete the 'b = a' operation first.
*		b(2, 4) = 5;				// Host-side write, already synchronized.
*
*	Although host-side transfers aren't usually asynchronous, it would be efficient to use DMA here if
*	possible.
*
*	This example demonstrates that both side of a pending asynchronous operation need to be aware of
*	the operation because 'a' can't be written to while asynchronously being pulled from and 'b' can't be
*	written to while asynchronously filling in from the copy operation.  Both 'a' and 'b' need to know
*	to wait and synchronize before executing the single-element modification operation.
*
*	As another example:
* 		auto c = ndarray<float>::zeros({32, 32}, host_pinned);
*		auto my_stream = GPUStream::New();
*		c.to_device(my_stream);
*		cout << "c @ 5,5 = " << to_string(c(5,5)) << endl;
*
*	In this case, we initiate an asynchronous transfer from the host_pinned buffer of 'c' to a GPU
*	stream.  We then read a value from 'c' host-side.  There is no need for the read operation to
*	block until c.to_device() completes as the 'c' host-side buffer is not being modified by the
*	memory transfer.
*
*	Asynchronous Implementation _____
*
*	We can see from the above examples that individual buffers need to track pending asynchronous
*	operations and need to differentiate read-blocking vs write-blocking asynchronous operations.
*
*	When synchronization is necessary, we want to avoid several independent synchronizations being
*	initiated on the same stream.  This can be accomplished with the GPUStream TrackAsync() 
*	mechanism.
*/

#ifndef __WB_ndarray_memory_h__
#define __WB_ndarray_memory_h__

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../System/GPU.h"
#include "ndarray_definitions.h"
#include "ndarray_helpers.h"

namespace wb { namespace math { namespace memory {	

	#pragma region "NDArrayBuffer class"

	class NDArrayBuffer : public cuda::IGPUStreamAsyncWatcher
	{		
	protected:		
		
		/// <summary>
		/// The alignment vector can have any length and specifies the alignment requirements 
		/// for any memory to be allocated.  See the calculate_strides() description to understand 
		/// the format of the alignment vector.
		///
		/// Design notes: the alignment might work better as a template parameter, but that isn't 
		/// supported by C++11.  It is in C++20.  But, "better" was a thought early in this design 
		/// process and may not hold up.  The alignment isn't used by a few of the descendant 
		/// classes, but it is used in enough of them that it's convenient to place it here in 
		/// the common ancestor class.
		/// </summary>
		std::vector<size_t> alignment;

		/// <summary>
		/// If equal to GPUStream::None(), then no asynchronous read is pending.  Otherwise, an asynchronous
		/// read is outstanding from this buffer.  This buffer will be notified by a Synchronize() call in
		/// the GPUStream and p_pending_async_read will be reset to GPUStream::None() at that time.  This
		/// member is managed by calling the add_pending_async_read() and/or finish_reads() functions.  It
		/// is important that any asynchronous reads from this buffer result in a call to 
		/// add_pending_async_read() so that synchronization is applied properly when necessary.
		/// </summary>
		cuda::GPUStream p_pending_async_read;

		/// <summary>
		/// If equal to GPUStream::None(), then no asynchronous write is pending.  Otherwise, an asynchronous
		/// write is outstanding to this buffer.  This buffer will be notified by a Synchronize() call in
		/// the GPUStream and p_pending_async_write will be reset to GPUStream::None() at that time.  This
		/// member is managed by calling the add_pending_async_write() and/or finish_write() functions.  It
		/// is important that any asynchronous writes to this buffer result in a call to 
		/// add_pending_async_write() so that synchronization is applied properly when necessary.
		/// </summary>
		cuda::GPUStream p_pending_async_write;

		cuda::GPUStream&& transfer(cuda::GPUStream&& stream, NDArrayBuffer& from_buffer)
		{
			#ifdef CUDA_Support
			if (!stream.IsNone())
			{
				stream.StopTrackAsync(&from_buffer);
				stream.TrackAsync(this);
			}
			#endif
			return std::move(stream);
		}

		void on_async_completed(cuda::GPUStream& stream) override
		{
			#ifdef CUDA_Support
			if (stream == p_pending_async_read) p_pending_async_read = cuda::GPUStream::None();
			if (stream == p_pending_async_write) p_pending_async_write = cuda::GPUStream::None();
			#endif
		}

	public:		
		/// <summary>
		/// This flag is managed by the owning ndarray when a different buffer for the same ndarray has been written 
		/// to.  When that happens, the other buffer becomes the up-to-date copy and all other buffers are 
		/// outdated.  The other buffers can retain their memory and stay allocated because it is common to 
		/// move data between buffers (i.e. between host and device memory).  Extra calls could release the 
		/// memory explicitly if desired, but that would be a non-automatic optimization.
		/// 
		/// The outdated flag is managed by the owning ndarray but usually starts out marked "true" since data is
		/// not yet allocated.  The exception is any of the ExistingBuffer classes, which sets outdated to false
		/// in some cases.  Allocating does not alter this flag and the owning ndarray must set it.
		/// </summary>
		bool outdated;				

		NDArrayBuffer(const vector<size_t>& alignment_)
			: alignment(alignment_), 
			outdated(true), 
			p_pending_async_read(cuda::GPUStream::None()),
			p_pending_async_write(cuda::GPUStream::None())
		{
		}

		NDArrayBuffer(NDArrayBuffer&& mv) noexcept
			:
			outdated(mv.outdated),
			p_pending_async_read(cuda::GPUStream::None()),
			p_pending_async_write(cuda::GPUStream::None())
		{
			mv.outdated = true;
			this->p_pending_async_read = transfer(std::move(mv.p_pending_async_read), mv);
			this->p_pending_async_write = transfer(std::move(mv.p_pending_async_write), mv);
		}

		NDArrayBuffer& operator=(NDArrayBuffer&& mv)
		{
			this->outdated = mv.outdated; mv.outdated = true;
			this->p_pending_async_read = transfer(std::move(mv.p_pending_async_read), mv);
			this->p_pending_async_write = transfer(std::move(mv.p_pending_async_write), mv);
			return *this;
		}

		virtual bool is_host_memory() const = 0;
		virtual bool is_device_memory() const { return !this->is_host_memory(); }		

		virtual ~NDArrayBuffer() { 
			#ifdef CUDA_Support
			if (!p_pending_async_read.IsNone()) p_pending_async_read.StopTrackAsync(this);
			if (!p_pending_async_write.IsNone()) p_pending_async_write.StopTrackAsync(this);
			#endif
			free();
		}

		virtual bool is_allocated() const = 0;

		virtual void free() { }

		virtual vector<size_t> calculate_strides(const ndarray_allocation_t& requirements) {
			return ::wb::math::memory::calculate_strides(this->alignment, requirements.shape, requirements.element_size);
		}

		/// <summary>
		/// Indicates whether this buffer type can allocate memory or is constrained to pre-existing
		/// memory.  Outstanding asynchronous operations are not considered by this function, but 
		/// it is assumed that allocate() can resolve them as needed before proceeding.  The
		/// buffer must be writable after allocation.
		/// </summary>
		virtual bool can_allocate() const {
			return false;
		}

		/// <summary>
		/// Allocates the memory requested within the context.  Throws
		/// an exception upon failure.  If memory is already allocated, then
		/// it can be reallocated to the new purpose if that is possible.  Otherwise,
		/// the old memory should be freed and a new buffer allocated.  The buffer
		/// must be writable after allocation.
		/// </summary>		
		virtual void allocate(const ndarray_allocation_t& requirements, vector<size_t>& need_strides) = 0;

		/// <summary>
		/// The can_write() function checks whether this buffer will support writing.  It must
		/// return true even before the buffer has been allocated and should remain writable (or not)
		/// for the lifetime of the buffer.
		/// </summary>
		/// <returns></returns>
		virtual bool can_write() const {
			return false;
		}

		/** Asynchronous tracking **/

		bool is_pending_async_read() const {
			return !p_pending_async_read.IsNone();
		}

		bool is_pending_async_write() const {
			return !p_pending_async_write.IsNone();
		}

		void add_pending_async_read(cuda::GPUStream stream)
		{
			if (stream.IsNone()) return;

			// Check if a different stream is reading from us or writing to us already.  
			// We'll have to let that one finish before switching to a new stream.
			if (!p_pending_async_read.IsNone() && p_pending_async_read != stream)
				p_pending_async_read.Synchronize();
			if (!p_pending_async_write.IsNone() && p_pending_async_write != stream)
				p_pending_async_write.Synchronize();

			// Calling Synchronize() above should cause the IGPUStreamAsyncWatcher notification to trigger,
			// and that should, in turn, set p_pending_async_xxx to None.
			assert(p_pending_async_read.IsNone() || p_pending_async_read == stream);

			p_pending_async_read = stream;
		}

		void add_pending_async_write(cuda::GPUStream stream)
		{
			if (stream.IsNone()) return;

			// Check if a different stream is reading from us or writing to us already.  
			// We'll have to let that one finish before switching to a new stream.
			if (!p_pending_async_read.IsNone() && p_pending_async_read != stream)
				p_pending_async_read.Synchronize();
			if (!p_pending_async_write.IsNone() && p_pending_async_write != stream)
				p_pending_async_write.Synchronize();

			// Calling Synchronize() above should cause the IGPUStreamAsyncWatcher notification to trigger,
			// and that should, in turn, set p_pending_async_xxx to None.
			assert(p_pending_async_write.IsNone() || p_pending_async_write == stream);

			p_pending_async_write = stream;
		}

		/// <summary>
		/// Blocks until any pending reads from this buffer have completed.
		/// </summary>
		void finish_reads()
		{
			if (!p_pending_async_read.IsNone())
				p_pending_async_read.Synchronize();
			assert(p_pending_async_read.IsNone());
		}

		/// <summary>
		/// Blocks until any pending writes to this buffer have completed.
		/// </summary>
		void finish_writes()
		{
			if (!p_pending_async_write.IsNone())
				p_pending_async_write.Synchronize();
			assert(p_pending_async_write.IsNone());
		}
	};
	
	#pragma endregion

	#pragma region "Host Buffers"

	class HostBuffer : public NDArrayBuffer
	{
		typedef NDArrayBuffer base;			

	public:		
		HostBuffer(const vector<size_t>& alignment_) : base(alignment_)
		{
		}

		HostBuffer(HostBuffer&& mv) : base(std::move(mv)) { }
		HostBuffer& operator=(HostBuffer&& mv) {
			base::operator=(std::move(mv));
			return *this;
		}

		bool is_host_memory() const override { return true; }

		virtual void* get_host_pointer() = 0;
	};
	
	class RegularHostBuffer : public HostBuffer
	{
		typedef HostBuffer base;				

	protected:
		void* p_data;		

		/// <summary>
		/// Indicates the allocated number of bytes addressed by p_data.
		/// </summary>
		size_t allocated_size;

		/// <summary>
		/// The is_reusable() function tests whether the existing allocation of this buffer can
		/// support the required parameters without needing to reallocate.
		/// </summary>
		bool is_reusable(const size_t need_size, size_t need_pointer0_alignment)
		{
			if (this->p_data == nullptr) return false;			
			if (need_size > this->allocated_size) return false;
			// Lastly, verify that the starting address of the buffer fits the alignment criteria.
			// The largest alignment requirement would always be the first entry in alignment, 
			// and so if that one is met then all are met.  
			return ((uintptr_t)p_data % need_pointer0_alignment) == 0;
		}

	public:		
		RegularHostBuffer(std::vector<size_t> alignment_ = std::vector<size_t>{ 16 /*bytes*/ })
			: p_data(nullptr), base(alignment_)
		{
		}

		RegularHostBuffer(RegularHostBuffer&& mv) noexcept : base(std::move(mv))
		{
			// Slightly faster than calling operator=(), because operator=() has to call free().
			this->p_data = mv.p_data; mv.p_data = nullptr;
			this->allocated_size = mv.allocated_size; mv.allocated_size = 0;
		}

		RegularHostBuffer& operator=(RegularHostBuffer&& mv)
		{
			free();
			base::operator=(std::move(mv));
			this->p_data = mv.p_data; mv.p_data = nullptr;			
			this->allocated_size = mv.allocated_size; mv.allocated_size = 0;
			return *this;
		}

		void free() override
		{
			if (this->p_data != nullptr)
			{				
				_aligned_free(this->p_data);				
				this->p_data = nullptr;
			}
			this->allocated_size = 0;
		}

		bool is_allocated() const override { return this->p_data != nullptr; }

		/// <summary>
		/// Indicates whether this buffer can allocate memory or is constrained to pre-existing
		/// memory.
		/// </summary>				
		bool can_allocate() const override {
			return true;
		}

		/// <summary>
		/// Allocates or reallocates the memory requested within the 
		/// context.  Throws an exception upon failure.
		/// </summary>
		void allocate(const ndarray_allocation_t& requirements, vector<size_t>& need_strides) override
		{
			auto need_size = calculate_required_size(need_strides, requirements.shape);
			size_t need_pointer0_alignment = (this->alignment.size() >= 1) ? this->alignment[0] : 1;
			if (is_reusable(need_size, need_pointer0_alignment)) return;
			
			if (this->p_data != nullptr)
			{
				/** can reshape instead of allocating **/
												
				this->p_data = _aligned_realloc(this->p_data, need_size, need_pointer0_alignment);
				// The MS docs on _aligned_realloc() are a bit unclear on error conditions.  It indicates
				// that it can provide an errno, but not exactly when you can consider it an error and
				// therefore when the errno is correctly set by _aligned_realloc() as opposed to a previous
				// function call.  The following is a safe solution, but does not provide any detail to the 
				// error.
				if (this->p_data == nullptr) throw OutOfMemoryException();				
			}
			else
			{
				/** Allocate new memory **/

				this->p_data = _aligned_malloc(need_size, need_pointer0_alignment);
				if (this->p_data == nullptr) wb::Exception::ThrowFromErrno(errno);
			}
			this->allocated_size = need_size;
		}		

		/// <summary>
		/// Indicates whether this buffer can be written to.
		/// </summary>
		bool can_write() const override {
			return true;
		}
		
		void* get_host_pointer() override {
			return p_data;
		}
	};

	#pragma endregion

	#pragma region "CUDA Buffers"

	#ifdef CUDA_Support

	template<HostFlags flags>
	class CUDAHostBuffer : public RegularHostBuffer
	{
		typedef RegularHostBuffer base;
			
		void validate_flags()
		{
			if ((flags & HostFlags::portable) != 0
				|| (flags & HostFlags::mapped) != 0
				|| (flags & HostFlags::write_combined) != 0)
			{
				if (!(flags & HostFlags::pinned))
					throw NotSupportedException("Cannot apply portable, mapped, or write_combined flags unless Pinned is specified.");
			}
		}		

	public:
		CUDAHostBuffer(std::vector<size_t> alignment_ = std::vector<size_t>{ 16 /*bytes*/ })
			: base(alignment_)
		{
			validate_flags();
		}

		CUDAHostBuffer(CUDAHostBuffer&& mv) = default;
		CUDAHostBuffer& operator=(CUDAHostBuffer&& mv) = default;

		void free() override
		{
			if (!(flags & HostFlags::pinned))
				base::free();
			else
			{
				if (this->p_data != nullptr)
				{					
					cudaThrowable(cudaFreeHost(this->p_data));					
					this->p_data = nullptr;
				}
				this->allocated_size = 0;
			}
		}

		/// <summary>
		/// Indicates whether this buffer can allocate memory or is constrained to pre-existing
		/// memory.
		/// </summary>				
		bool can_allocate() const override {
			return true;
		}

		/// <summary>
		/// Allocates or reallocates the memory requested within the 
		/// context.  Throws an exception upon failure.
		/// </summary>
		void allocate(const ndarray_allocation_t& requirements, vector<size_t>& need_strides) override
		{
			if (!(flags & HostFlags::pinned))
			{
				base::allocate(requirements, need_strides);
				return;
			}			
			
			auto need_size = calculate_required_size(need_strides, requirements.shape);
			size_t need_pointer0_alignment = (this->alignment.size() >= 1) ? this->alignment[0] : 1;
			if (is_reusable(need_size, need_pointer0_alignment)) return;

			if (this->p_data != nullptr)
			{
				/** No API for reshaping in CUDA memory, and we've already
					checked that the required size is larger than the
					current size or otherwise incompatible.  So free the
					memory and allocate anew.  **/
				free();
			}

			/** Allocate new memory **/
			assert(this->p_data == nullptr);
			unsigned int cudaFlags = 0;
			if ((flags & HostFlags::portable) != 0) cudaFlags |= cudaHostAllocPortable;
			if ((flags & HostFlags::mapped) != 0) cudaFlags |= cudaHostAllocMapped;
			if ((flags & HostFlags::write_combined) != 0) cudaFlags |= cudaHostAllocWriteCombined;
			cudaThrowable(cudaHostAlloc(&this->p_data, need_size, cudaFlags));
			if (this->p_data == nullptr) throw OutOfMemoryException();
			this->allocated_size = need_size;
		}		

		/// <summary>
		/// Indicates whether this buffer can be written to.
		/// </summary>
		bool can_write() const override {
			return true;
		}
	};			

	class CUDADeviceBuffer : public NDArrayBuffer
	{
		typedef NDArrayBuffer base;

	protected:
		void* p_data;

		/// <summary>
		/// This flag indicates that the buffer was created with the "optimize strides" option and that strides need not be
		/// fixed to a particular alignment if a more optimal alignment can be identified for the GPU device.
		/// </summary>
		bool optimize_strides;

		/// <summary>
		/// The CUDADeviceBuffer class allows for "soft" reallocations where the dimensions of the image are changed within a
		/// buffer that is already allocated.  There are certain rules on when this is allowed.  This is particularly helpful
		/// when using a CUDADeviceBuffer as a reusable scratch buffer.  In order to facilitate these soft reallocations,
		/// we need to keep track of the true allocated stride and size.
		/// </summary>
		size_t allocated_size;

		/// <summary>
		/// The is_reusable() function tests whether the existing allocation of this buffer can
		/// support the required parameters without needing to reallocate.
		/// </summary>
		bool is_reusable(const size_t need_size, size_t need_pointer0_alignment)
		{
			if (this->p_data == nullptr) return false;
			if (need_size > this->allocated_size) return false;
			// Lastly, verify that the starting address of the buffer fits the alignment criteria.
			// The largest alignment requirement would always be the first entry in alignment, 
			// and so if that one is met then all are met.  
			return ((uintptr_t)p_data % need_pointer0_alignment) == 0;
		}

	public:			
		cuda::GPUStream next_stream;

		CUDADeviceBuffer(cuda::GPUStream stream, std::vector<size_t> alignment_)
			: p_data(nullptr), base(alignment_), next_stream(stream), optimize_strides(false)
		{
		}

		CUDADeviceBuffer(cuda::GPUStream stream)
			: p_data(nullptr), base(std::vector<size_t> {}), next_stream(stream), optimize_strides(true)
		{
		}

		CUDADeviceBuffer(CUDADeviceBuffer&& mv) = default;
		CUDADeviceBuffer& operator=(CUDADeviceBuffer&& mv) = default;

		void free() override
		{			
			if (this->p_data != nullptr)
			{
				// According to this page: https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
				// cudaFree() automatically snyhronizes the device before freeing memory and so this may be unnecessary.  However,
				// I am switching to cudaFreeAsync() which is a stream-oriented free operation.  It is unnecessary to wait for
				// pending operations on the stream unless a different stream needs to access the same memory.  For this
				// asynchronous version, we leave the is_pending flag set and it marks not only kernels operating on the
				// memory but the allocations/deallocations themselves.  Synchronization should happen before copying to/from the
				// stream with host or other streams.
				/*
				if (this->is_pending)
				{					
					#error revisit and be sure the pending marker is getting set...
					cudaThrowable(cudaStreamSynchronize(pending_stream));
					this->is_pending = false;
				}
				*/

				if (this->next_stream == 0)
				{
					finish_reads();
					finish_writes();					
					cudaThrowable(cudaFree(this->p_data));
				}
				else
				{
					cudaThrowable(cudaFreeAsync(this->p_data, this->next_stream));
					add_pending_async_write(this->next_stream);
				}
				this->p_data = nullptr;
				this->allocated_size = 0;
			}
		}

		bool is_allocated() const override { return this->p_data != nullptr; }

		bool is_host_memory() const override { return false; }		

		vector<size_t> get_alignment(const ndarray_allocation_t& requirements)
		{
			if (requirements.shape.size() > 1 && this->optimize_strides)
			{
				// Optimize the pitch (row stride) in particular, if this is a 2+D array.
				size_t need_minimum_pitch = requirements.shape[requirements.shape.size() - 1] * requirements.element_size;
				size_t optimal_pitch = (size_t)this->next_stream.GetGSI().GetOptimalPitch(this->next_stream.GetDeviceId(), (int)need_minimum_pitch);

				// Reformulate this into the 'alignment' spec and run the base calculate_strides().
				auto alignment = vector<size_t>({ optimal_pitch, requirements.element_size });
				return alignment;
			}
			else return this->alignment;
		}

		vector<size_t> calculate_strides(const ndarray_allocation_t& requirements) override
		{
			return ::wb::math::memory::calculate_strides(get_alignment(requirements), requirements.shape, requirements.element_size);
		}

		/// <summary>
		/// Indicates whether this buffer can allocate memory or is constrained to pre-existing
		/// memory.
		/// </summary>
		bool can_allocate() const override {
			return true;
		}

		/// <summary>
		/// Allocates or reallocates the memory requested within the 
		/// context.  Throws an exception upon failure.
		/// </summary>
		void allocate(const ndarray_allocation_t& requirements, vector<size_t>& need_strides) override
		{
			auto need_size = calculate_required_size(need_strides, requirements.shape);
			auto alignment = this->get_alignment(requirements);
			size_t need_pointer0_alignment = alignment.size() > 0 ? alignment[0] : 1;
			if (is_reusable(need_size, need_pointer0_alignment)) return;

			if (this->p_data != nullptr)
			{
				/** No API for reshaping in CUDA memory, and we've already
					checked that the required size is larger than the
					current size or otherwise incompatible.  So free the
					memory and allocate anew.  **/
				free();
			}

			/** Allocate new memory **/
			assert(this->p_data == nullptr);
			// There are routines such as cudaMallocPitch() or cudaMalloc3D().  I am assuming that these functions insert
			// pitch using calculations similar to what I have here.
			if (this->next_stream == 0)
			{
				finish_reads();
				finish_writes();
				cudaThrowable(cudaMalloc(&this->p_data, need_size));
			}
			else
			{
				// Note: cudaMallocAsync() operates using memory pools.  There may be optimization to be had from manipulating
				// thresholds regarding releasing memory from pools and such.  See the article here for more details:
				//	https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/
				cudaThrowable(cudaMallocAsync(&this->p_data, need_size, this->next_stream));
				add_pending_async_write(this->next_stream);
			}
			if (this->p_data == nullptr) throw OutOfMemoryException();
			if (((uintptr_t)this->p_data % need_pointer0_alignment) != 0)
				throw NotSupportedException("Expected CUDA to provide memory allocation starting on a device addressed aligned to requested needs.");						
			this->allocated_size = need_size;
		}		

		/// <summary>
		/// Indicates whether this buffer can be written to.
		/// </summary>
		bool can_write() const override {
			return true;
		}

		void* get_device_pointer() {
			return p_data;
		}
	};

	#endif

	#pragma endregion		

	#pragma region "FreeImage compatibility"
	#ifdef FreeImage_Support
	namespace FI
	{
		inline FREE_IMAGE_TYPE fi_type_from_ID(ElementTypeID id)
		{
			switch (id)
			{						
			case ElementTypeID::Int16: return FIT_INT16;
			case ElementTypeID::Int32: return FIT_INT32;
			case ElementTypeID::UInt8: return FIT_BITMAP;
			case ElementTypeID::UInt16: return FIT_UINT16;
			case ElementTypeID::UInt32: return FIT_UINT32;
			case ElementTypeID::Float32: return FIT_FLOAT;
			case ElementTypeID::Float64: return FIT_DOUBLE;
			case ElementTypeID::Complex64x2: return FIT_COMPLEX;
			case ElementTypeID::RGB24: return FIT_BITMAP;
			case ElementTypeID::RGBA: return FIT_BITMAP;
			default: throw NotSupportedException("The requested element type is not supported by the FreeImage API.");
			}
		}				

		inline void ErrorHandler(FREE_IMAGE_FORMAT fif, const char* message) {
			std::string ret = "Image error";
			if (fif != FIF_UNKNOWN) {
				ret += " (" + std::string(FreeImage_GetFormatFromFIF(fif)) + " format): ";
			}
			else ret += ": ";
			ret += std::string(message);
			throw IOException(ret);
		}
	}
	
	class FreeImageHostBuffer : public HostBuffer
	{
		typedef HostBuffer base;

		/// <summary>
		/// The FreeImage handle to the memory buffer.
		/// </summary>
		FIBITMAP* p_fi;

	public:
		FreeImageHostBuffer()
			: p_fi(nullptr), base(std::vector<size_t>({}))
		{
		}		

		void free() override
		{
			if (this->p_fi != nullptr) { FreeImage_Unload(this->p_fi); this->p_fi = nullptr; }
		}

		FreeImageHostBuffer(FreeImageHostBuffer&& mv) noexcept : base(std::move(mv))
		{						
			this->p_fi = std::move(mv.p_fi);
			mv.p_fi = nullptr;
		}

		FreeImageHostBuffer& operator=(FreeImageHostBuffer&& mv)
		{
			free();
			base::operator=(std::move(mv));
			this->p_fi = mv.p_fi; mv.p_fi = nullptr;
			return *this;
		}

		#if 0
		#ifdef FreeImage_Support
		/// <summary>
		/// Creates a new HostImageData object that takes responsibility for a FreeImage memory buffer and establishes it for use
		/// as an Image data buffer.  The data is not copied but the HostImageData becomes responsible for using FreeImage_Unload()
		/// on the FIBITMAP* when it is no longer needed.
		/// </summary>								
		static HostImageData NewOwner(FIBITMAP* pFIB)
		{
			auto ret = HostImageData(HostFlags::None);
			ret.m_Width = FreeImage_GetWidth(pFIB);
			ret.m_Height = FreeImage_GetHeight(pFIB);
			ret.m_Stride = FreeImage_GetPitch(pFIB);
			ret.m_pData = FreeImage_GetBits(pFIB);
			// See also notes on the m_pFileData member.
			ret.m_Responsibility = DataResponsibility::RdWr;
			ret.m_pFileData = pFIB;
			return ret;
		}
		#endif
		#endif

		/// <summary>
		/// Allocates or reallocates the memory requested within the 
		/// context.  Throws an exception upon failure.
		/// </summary>
		void allocate(const ndarray_allocation_t& requirements, vector<size_t>& need_strides) override
		{
			// FreeImage only operates on 2D images
			if (requirements.shape.size() != 2)
				throw NotSupportedException("FreeImageHostBuffer can only operate on 2D buffers.");
						
			free();			

			/** Allocate new memory **/				

			if (this->p_fi != nullptr) throw Exception("Expected p_fi to be null.");
			FreeImage_SetOutputMessage(FI::ErrorHandler);

			auto width = requirements.shape[1];
			auto height = requirements.shape[0];
			int bpp = bits_per_element_from_ID(requirements.element_type);
			this->p_fi = FreeImage_AllocateT(FI::fi_type_from_ID(requirements.element_type), (int)width, (int)height, bpp);
			if (this->p_fi == nullptr) throw OutOfMemoryException();		
			need_strides = vector<size_t>({ FreeImage_GetPitch(this->p_fi), (size_t)(bpp / 8) });
		}

		void* get_host_pointer() override {
			return FreeImage_GetBits(this->p_fi);
		}
	};
	
	#endif	

	#pragma endregion

	#pragma region "Existing buffers"

	// TODO: ExistingBuffers
	// TODO: these should perhaps be accessible in the wb::math namespace without needing to say memory::ReadonlyExistingBuffer.  Or aliased.
	// This part gets a bit tricky.  The caller will directly instantiate these classes.  They could be based on RegularHostBuffer or FreeImageHostBuffer,
	// with adding a DataResponsibility template parameter, but how would it select between the two?  FreeImageHostBuffer only applies for a 2D case.
	// It could be a wrapper that has two pointers, one to a RegularHostBuffer and one to FreeImageHostBuffer and only one gets used at a time.
	// TODO: set the outdated flag false when initializing the 'initial_data' flag (default) and otherwise to true.

	#pragma endregion	

	}// end the 'memory' namespace but still in wb::math

	#pragma region "Selector factories where needed"

	#ifdef FreeImage_Support
	global_variable memory::GenericNDArrayAllocator<memory::FreeImageHostBuffer> FreeImage_stream;
	global_variable memory::GenericNDArrayAllocator<memory::RegularHostBuffer> nonimage_stream;	
	#endif

	#pragma endregion

	#pragma region "Constants"

	/// <summary>
	///	The host_contiguous stream can be utilized if you specifically need to allocate
	/// buffers where the elements are stored contiguously.  By default, buffers use
	/// aligned memory that is usually optimal for processing and memory access instead.
	/// </summary>
	global_variable memory::GenericNDArrayAllocator<memory::RegularHostBuffer, 1> host_contiguous;

	#ifdef CUDA_Support	
	global_variable memory::GenericNDArrayAllocator<memory::CUDAHostBuffer<memory::HostFlags::pinned>> host_pinned;
	global_variable memory::GenericNDArrayAllocator<memory::CUDAHostBuffer<constexpr_or(memory::HostFlags::pinned, memory::HostFlags::portable)>> host_portable;
	#endif
	
	#if defined(CUDA_Support)
	global_variable memory::GenericNDArrayAllocator<memory::CUDAHostBuffer<memory::HostFlags::none>> host_stream;
	#else
	global_variable memory::GenericNDArrayAllocator<memory::RegularHostBuffer> host_stream;
	#endif

	// Want to avoid defining 'gpu_stream' or 'device_stream' that chooses any stream because 
	// GPUStream::None() leads to not having access to a global GPUSystemInfo that can provide 
	// the optimal pitch needed by allocate().  This enforces that we must always specify a
	// particular GPUStream, which is good for performance but not the simplest usage.  But, we are
	// in C++ and not Python so performance outweighs simplicity here.
	
	// TODO: probably want to add a way to write 'ContiguousBuffer(my_stream)' in order to override the
	// optimal alignment in favor of contiguous memory for special cases where it is needed.

	#pragma endregion

} }		// end namespaces

#endif	// __WB_ndarray_memory_h__

//	End of ndarray_memory.h

