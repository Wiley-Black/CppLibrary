/////////
//	NDArray.h
/////
//	Design Notes:
/*
	This library follows the design patterns of numpy's NDArray type and everything that flows forth from that, though with additional
	functionality such as GPU support built-in.  This C++ library avoids sacrificing performance or at least performance potential,
	and so it has more options than the Python equivalent.  For example, fine-grained control over memory allocation behaviors is
	possible.

	In Python, one can allocate a new ndarray several ways:

		my_array = np.ndarray((3, 64, 64), dtype=float)
		my_array = np.zeros((3, 64, 64), dtype=int)
		my_array = np.asarray([[11, 12, 13], [21, 22, 23]], device="cpu")

	The last one being an interesting new addition in numpy 2 that seems like a path toward GPU support that isn't there yet.
	In Numpy 2, a .to_device() function is also noticed but not documented.

	One can use cupy to allocate a similar array on the GPU device:

		my_array = cupy.ndarray((3, 64, 64), dtype=float)

	Some necessary differences here begin with dtype.  Since C++ is strongly typed, we will want to specify the dtype as a template
	argument.  Thus we might have:

		using namespace wb::math;
		my_array = ndarray<float>({3, 64, 64});
		my_array = ndarray<float>::zeros({3, 64, 64});
		my_array = ndarray<float>::asarray({{11, 12, 13}, {21, 22, 23}});

	Not yet addressed is CPU vs. GPU functionality.  The ndarray must track what device(s) it is stored upon.  A simple solution
	might be to specify the device via a string or enumeration, such as adding a "device" argument as Python's Numpy 2 has done.
	However, memory allocation can have a lot more options to it than Python would probably expose.  For example, we might
	want to allocate host memory for an array, fill it on the host, and then transfer it to the GPU device.  This can require
	allocating the host memory with specific flags and using the CUDA host memory allocation functions.  

	Here, I define a context or "stream" instead of just a device.  One can put context to an image such as "destined 
	for the GPU" or more directly, "on the host but should be pinned".  By keeping this abstract, different allocation methods 
	can be supported and image stream(s) can be tracked in arbitrary (and thus extensible) fashion.  CUDA already does this by
	defining data streams, so we need only extend that to include all ndarray memory allocation functionality related to the
	particular image.

		using namespace wb::math;
		my_array = ndarray<float>({3, 64, 64});						// Default stream: host device with no GPU expectation.
		my_array = ndarray<float>({3, 64, 64}, host_pinned);		// Allocates in host, pinned memory.		

		using namespace wb::cuda;
		my_stream = GPUStream();
		my_array = ndarray<float>({3, 64, 64}, my_stream);			// Allocates on specific gpu device stream.

	This is made possible by the cpu_pinned and gpu stream constants, as well as GPUStream being a compatible descendant 
	of the stream class.  It is also possible to re-use existing memory without taking responsibility for that memory:

		void* rw_memory_buffer = ...;
		void* ro_memory_buffer = ...;
		{
			// Define a writable ndarray of shape 3x64x64 from a "tight" memory buffer that already exists:
			auto my_array_rw = ndarray<float>({3, 64, 64}, WritableExistingHostBuffer(memory_buffer_rw, {64*64, 4}));

			// Define a read-only ndarray of shape 3x64x64 from an existing memory buffer that is not "tight":
			auto my_array_ro = ndarray<float>({3, 64, 64}, ReadOnlyExistingHostBuffer(memory_buffer_ro, {96*64, 4}));

			// These variables need to go out of scope before we free their underlying memory, as shown here.
		}
		free(rw_memory_buffer);
		free(ro_memory_buffer);

	Arrays can also be moved between devices or exist on more than one device in more than one state.  For example:
		my_array = ndarray<float>::zeros({2, 4}, host_pinned);			// Allocates on host device in pinned memory.
		my_array.to_device();											// Array now exists on both CPU and GPU, with modifications applied at the GPU.
		my_array += 1;													// After this operation, the array is now outdated on the CPU.
		cout << "Result of first pixel: " << my_array[0][0] << end;		// CPU copy is outdated, so this retrieves the GPU value and prints "1".

	The .to_device() function can also accept a specific GPUStream.  Some functions will throw an exception under certain 
	circumstances relating to the stream, such as a attempting to use += on an array that was setup using a ReadOnlyExistingHostBuffer 
	context.  Attempting to modify a ReadOnlyExistingHostBuffer in-place always results in an error.  However, new arrays can be created 
	from ReadOnlyExistingHostBuffer ndarrays as long as they are not created in-place.  For example,

		new_array = my_array + 1;

	There are also cases where the caller may want to specify multiple streams.  For instance:

		auto my_stream = GPUStream();
		auto my_array_ro = ndarray<float>({3, 64, 64}, { ReadOnlyExistingBuffer(memory_buffer_ro, {3, 64, 64}), my_stream });

	In this case, the read-only existing buffer is utilized initially.  A later operation such as,

		my_array_ro.to_device();

	would then examine the available stream list and decide that the ReadOnlyExistingHostBuffer is not a device stream.  It will
	proceed to utilize my_stream and copy from the read-only host memory to the device stream.  After this step, the host and 
	device memory are both considered up-to-date.  Reads can be made from either one without penalty.  A modification will happen 
	on the last location requested, and so:

		auto new_array = my_array_ro + 2;

	Would result in new_array being created as a new device-based ndarray that exists with the my_stream context.

	Design Notes _____

	The ndarray class heirarchy is built of 3 key levels: the buffer(s) (NDArrayBuffer-derived classes), the view(s) 
	(NDArrayBufferView), and the ndarray itself.  Each ndarray contains an array of view(s).  Each view couples with
	exactly one buffer, although buffers may be reused amongst multiple views.

	Interpretation of array data requires all 3 levels.  The buffer allocation is managed at the lowest level 
	(NDArrayBuffer).  The view tracks the memory layout of the buffer.  The ndarray tracks the shape and element
	type of the buffer.  

	Separation of the array metadata into these 3 levels allows flexibility.  One buffer can supply multiple
	views without needing to allocate more memory in many cases, and thus interpretation must pass through a view.
	An ndarray itself is always a kind of view.  Since ndarray objects only reference buffers through views, the
	creation of a new ndarray can represent either new buffer(s) or new view(s) into existing buffers.
*/
/////////

#ifndef __WB_ndarray_h__
#define __WB_ndarray_h__

#include "../wbFoundation.h"
#include "Implementation/ndarray_view.h"

namespace wb
{
	namespace math
	{		
		template<typename ElementType> class ndarray: public wb::diagnostics::DebugValidObject, public ndarray_allocation_t
		{
			typedef DebugValidObject base;
		
			#pragma region "Internal"

		protected:

			/// <summary>
			/// Provides the index of the view within the 'views' array that is
			/// active.  That is, in the case where multiple buffers are up-to-date, 
			/// which one was used most recently?  i.e. After a to_device() call, 
			/// the device buffer should be utilized for the next write operation.  Read 
			/// operations can ignore the active_index and will prefer the optimal buffer 
			/// for reads (i.e. host-side and not 'outdated' might be optimal).  However, 
			/// once a write occurs, all other buffers should be marked outdated.  The
			/// active_index is initially (size_t)-1, indicating that no buffer has been
			/// initialized for writing yet.  The active buffer must always be !outdated,
			/// and thus a writable and up-to-date buffer is required when active_index
			/// is set.
			/// </summary>
			size_t	active_index;

		protected:
			/// <summary>
			/// See the ndarray_memory.h for more discussion about NDArrayBuffer.  In addition, there are aspects for which
			/// the ndarray is responsible such as:
			///  - requesting allocation as needed,
			///	 - managing the 'outdated' flag *, 
			///  - being aware of pending asynchronous operations,
			///	 - and knowing which buffer is the "preferred write" target even if none are yet outdated.
			/// 
			/// The 'outdated' flag should be set on a buffer when a different buffer is the target of a write.
			/// Anytime that a buffer is 'outdated', then can_use_as() must be checked again before attempting 
			/// use and if false then can_allocate() must be checked as well (followed by an allocate() if true).
			/// 
			/// Reshape operations, typecast-in-place, slicing, and transpose operations can be facilitated through
			/// the use of views.  Views provide an alternative access pattern into the same underlying buffer.
			/// Each buffer has at least one view on it, which would usually start out as a view into the full
			/// buffer as initially allocated.  Each view has a shared_ptr to an underlying buffer, and multiple
			/// ndarray objects can share the same underlying buffer(s) through different views.
			/// </summary>
			vector<unique_ptr<memory::NDArrayBufferView>>	views;			

			static void initiate_view_copy_async(memory::NDArrayBufferView& dst, memory::NDArrayBufferView& src, const vector<size_t>& shape, size_t element_size)
			{
				// Precondition: the views must have the same shape.  This is automatically true for
				// views/buffers within the same ndarray but may not apply to views/buffers for different 
				// ndarrays and the caller must address this before the copy.

				// This function can be used for intra or inter-ndarray copies with some constraints:
				//	1) caller is responsible for managing the outdated flag (except that the src must
				//		not be outdated at call time).
				//	2) views/buffers must have a common shape and element size.

				if (src.outdated)
					throw ArgumentException("Unable to initiate a copy from a buffer that is marked as outdated.");
				if (!src.is_allocated() || dst.is_allocated())
					throw ArgumentException("Buffers must be allocated prior to initiating buffer copy.");

				memory::copy_kind kind;
				if (src.is_host_memory())
				{
					if (dst.is_host_memory()) kind = memory::copy_kind::host_to_host; else kind = memory::copy_kind::host_to_device;
				}
				else
				{
					if (dst.is_host_memory()) kind = memory::copy_kind::device_to_host; else kind = memory::copy_kind::device_to_device;
				}

				cudaStream_t stream = (cudaStream_t)0;
				if (src.is_device_memory() || dst.is_device_memory())
				{
					GPUStream gpu_stream = ((memory::CUDADeviceBuffer&)src).next_stream;
					if (gpu_stream.IsNone()) gpu_stream = ((memory::CUDADeviceBuffer&)dst).next_stream;
					if (gpu_stream.IsNone())
					{
						src.finish_writes();
						dst.finish_reads();
					}
					else
					{
						src.add_pending_async_read(gpu_stream);
						dst.add_pending_async_write(gpu_stream);
						stream = (cudaStream_t)gpu_stream;
					}
				}

				void* p_src = src.get_pointer();
				void* p_dst = dst.get_pointer();
				memory::move_memory_async(p_dst, dst.strides, p_src, src.strides, shape, element_size, kind, stream);
			}

			void init(vector<size_t> shape, initializer_list<memory::INDArrayAllocator&> allocators)
			{
				// TODO optimization: buffer alignment could probably be optimized better.  The CUDA
				// device buffers identify the optimal pitch for the device and utilize that.
				// The host buffers?  They use a default value for alignment, but in many cases it
				// would be more optimal to have them use the same alignment as the CUDA device
				// buffers so that host <-> device memory transfers can transfer the entire image
				// in one block.

				this->active_index = (size_t)-1;
				this->shape = shape;
				for (auto allocator : allocators)
				{
					views.push_back(
						memory::NDArrayBufferView(
							std::shared_ptr<memory::NDArrayBuffer>(allocator.new_ndarray_buffer())
						));
					// If a WritableExistingBuffer containing initial data was added to the views list, 
					// then it should immediately become the active buffer.
					if (active_index != (size_t)-1) continue;
					auto& new_view = *views[views.size() - 1];
					if (new_view.is_allocated() && !new_view.is_outdated()
						&& new_view.can_write())
						active_index = buffers.size() - 1;
				}

				// If an active_index was found above, then no need to allocate one.
				if (active_index != (size_t)-1) return;

				// Determine if there is a writable buffer and allocate a buffer for it.  While I could set it up
				// such that we do not allocate until the first write occurs, that makes it more difficult to
				// allocate memory in advance and can cause first-loop timing to have different timings.  If the
				// caller intended a read-only buffer, then they needn't provide any writable buffers.				
				for (size_t ii = 0; ii < views.size(); ii++)
				{
					if (views[ii].can_write() && views[ii].can_allocate())
					{
						views[ii].allocate(*this);
						active_index = ii;
						return;
					}
				}
			}

			#pragma endregion
			#pragma region "Construction"

		public:

			vector<size_t> shape;
			
			ndarray(vector<size_t> shape, memory::INDArrayAllocator& allocator = memory::host_stream) {
				init(shape, { allocator });
			}

			ndarray(vector<size_t> shape, initializer_list<memory::INDArrayAllocator> allocators) {
				init(shape, allocators);
			}

			ndarray(ndarray&) = delete;
			ndarray& operator=(ndarray&) = delete;

			ndarray(ndarray<PixelType, FinalType>&& mv) noexcept
				: base(std::move(mv))
			{
				shape = std::move(mv.shape);
				views = std::move(mv.views);
				active_index = mv.active_index;
			}

			ndarray& operator=(ndarray&& mv)
			{								
				base::operator=(std::move(mv));
				shape = std::move(mv.shape);
				views = std::move(mv.views);
				active_index = mv.active_index;
				return *this;
			}

			#pragma endregion

			#pragma region "Initialization with values"		

			static ndarray zeros(vector<size_t> shape, initializer_list<memory::INDArrayAllocator> allocators) {
				ndarray result(shape, allocators);
				NDArrayBufferView& view = result.get_writable_view();				
				view.add_pending_async_write(view.get_next_stream());
				memory::initial_fill_memory_async(view.get_pointer(), view.strides, shape, (ElementType)0, view.is_host_memory(), view.get_next_stream());
				return result;
			}

			static ndarray zeros(vector<size_t> shape, memory::INDArrayAllocator& allocator = memory::host_stream) {
				return zeros(shape, { allocator });
			}

			static ndarray ones(vector<size_t> shape, initializer_list<memory::INDArrayAllocator> allocators) {
				ndarray result(shape, allocators);
				NDArrayBufferView& view = result.get_writable_view();
				view.add_pending_async_write(view.get_next_stream());
				memory::initial_fill_memory_async(view.get_pointer(), view.strides, shape, (ElementType)1.0, view.is_host_memory(), view.get_next_stream());
				return result;
			}

			static ndarray ones(vector<size_t> shape, memory::INDArrayAllocator& allocator = memory::host_stream) {
				return ones(shape, { allocator });
			}

			#pragma endregion

			#pragma region "View finders and access helpers"

		protected:

			enum class buffer_constraint
			{
				none,
				host,
				device
			};

			/// <summary>
			/// Retrieves the preferred view for reading.  Views that are marked as 'outdated' are
			/// ineligible.  Since the request is for reading, the "active_index" value is ignored.
			/// If no view is available for reading, an exception is raised.
			/// 
			/// If the returned view is host-side, then any writes to it are completed (synchronized) 
			/// before returning.  If the returned view is device-side then it might still have 
			/// asynchronous write operations pending.
			/// </summary>
			size_t get_readable_view_index(buffer_constraint requirement)
			{
				ValidateObject();

				// There are quite a few constraints as to buffer selection, enumerated here in order of requirement:
				//	1. If outdated and !can_allocate() then the buffer is ineligible.
				//	2. If buffer is outdated and no buffer can copy into it, it is ineligible.
				//	3. Buffer does not match caller's specified buffer_constraint requirement.
				// Buffer preferences:
				//	4. Prefer that it not be outdated.
				//  5. Prefer that the buffer memory already be allocated.				
				//	6. All other factors being equal, choose the buffer earliest in the buffers list.
				//
				// I could also check for "no pending asynchronous writes" that might delay the read,
				// but it seems like such a rare corner case to even have 3 buffers at all and then
				// to have 2 that are both eligible to transfer into prior to reading that I think it
				// might never come up.  And then whether you prefer the asynchronous writes or not
				// could be a guessing game as well.
				
				prepare = [](size_t view_index)
					{
						if (views[view_index]->is_host_memory()) views[view_index]->finish_writes();
						return view_index;
					};

				// First pass: in the interest of speed, check if there is a view already ready-to-go.
				for (size_t ii = 0; ii < views.size(); ii++)
				{
					if ((views[ii]->is_host_memory() && requirement == buffer_requirement::host)
						|| (views[ii]->is_device_memory() && requirement == buffer_requirement::device)
						|| requirement == buffer_requirement::none)
					{
						if (!views[ii]->is_outdated()) return prepare(ii);
					}
				}

				// Second pass: find a buffer that is outdated but otherwise fits the requirements and 
				// another buffer that can be copied from.
				size_t index_src = (size_t)-1, index_dst = (size_t)-1;
				bool dst_needs_allocate = false;
				for (size_t ii = 0; ii < views.size(); ii++)
				{
					if ((views[ii]->is_host_memory() && requirement == buffer_requirement::host)
						|| (views[ii]->is_device_memory() && requirement == buffer_requirement::device)
						|| requirement == buffer_requirement::none)
					{
						if (index_dst == (size_t)-1)
						{
							dst_needs_allocate = !views[ii]->is_allocated();
							if (!dst_needs_allocate || (dst_needs_allocate && views[ii]->can_allocate()))
								index_dst = ii;
						}
						else
						{							
							bool needs_allocate = !views[ii]->is_allocated();
							// we already have a candidate index_dst, and if needs_allocate is true at
							// this point then there is no advantage to this new buffer.  In accordance
							// with using the first-listed buffer, we don't use it.
							if (!needs_allocate)
							{
								index_dst = ii;
								dst_needs_allocate = false;
							}
						}
					}
					if (views[ii]->is_outdated()) continue;
					if (index_src == (size_t)-1) index_src = ii;
				}

				if (index_src < views.size() && index_dst < views.size())
				{
					// We found an up-to-date view that can be copied into another view that will meet 
					// the buffer_requirement.  Initiate a copy into the buffer.

					if (dst_needs_allocate) views[index_dst]->allocate(*this);
					initiate_view_copy_async(*views[index_dst], *views[index_src]);
					views[index_dst]->mark_updated();
					return prepare(index_dst);
				}

				throw IndexOutOfRangeException("No buffers are in a readable state.");
			}

			/// <summary>
			/// Retrieves the active buffer view.  The get_writable_view() call assumes that 
			/// writing will happen next and so it will mark all other views in this
			/// ndarray as outdated before returning.  
			/// 
			/// If the returned buffer view is a host-side buffer, then any outstanding 
			/// stream synchronization is performed before returning.  For a device-side 
			/// buffer, the returned buffer might still have asynchronous operations pending.
			/// </summary>
			/// <returns>The active buffer view that is ready to be modified.</returns>
			memory::NDArrayView& get_writable_view()
			{
				ValidateObject();
				if (active_index >= buffers.size())
					throw IndexOutOfRangeException("No buffer is available for writing.");				
													
				auto& view = *views[active_index];
				if (view.is_outdated())
					throw NotSupportedException("The active view should never be marked as outdated.");

				// Mark all views that are not the active_index as outdated at this time.
				for (size_t ii = 0; ii < views.size(); ii++)
					if (active_index != ii) views[ii].mark_outdated();

				if (view.is_host_memory()) 
				{
					view.finish_writes(); 
					view.finish_reads();
				}
				return view;
			}

#if 0			
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
#endif

			#pragma endregion
			#pragma region "Host/device transfer"

		public:			

			/// <summary>
			/// The to_host() call transfers image data to the host-side if it is not already 
			/// there.
			/// </summary>
			void to_host()
			{
				active_index = get_readable_view_index(buffer_constraint::host);
			}

			/// <summary>
			/// The to_device() call transfers image data to the device-side if it is not already
			/// available.  The operation proceeds asynchronously.
			/// </summary>
			void to_device()
			{
				#ifdef CUDA_Support
				active_index = get_readable_view_index(buffer_constraint::device);
				#else
				throw NotSupportedException();
				#endif
			}

			/// <summary>
			/// The to_stream() call will transition the device buffer onto the specified stream.
			/// The transition will happen asynchronously, but switching streams will necessitate
			/// synchronization as soon as a new operation is performed on the stream, if there
			/// are outstanding asynchronous operations.
			/// </summary>
			/// <param name="Stream">The new GPUStream to select for this ndarray</param>
			void to_stream(GPUStream stream)
			{
				#ifdef CUDA_Support
				for (auto& view : views)				
					if (is_type<CUDADeviceBuffer>(view.p_buffer))					
						dynamic_cast<CUDADeviceBuffer*>(view.p_buffer.get())->next_stream = stream;
				#endif
			}		

			#pragma endregion

			#pragma region "CUDA Helpers"
		protected:

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
			
			void	before_kernel_launch_aux(const char* pszSourceFile, int nLine)
			{
				// To check for kernel launch errors, we will require a call to 
				// cudaPeekAtLastError().  However we won't know if it is a launch
				// error from the launch we just made unless we clear any past errors.
				// cudaGetLastError() resets the host thread-based error variable to
				// cudaSuccess.
				::wb::cuda::Throwable(cudaGetLastError(), pszSourceFile, nLine);
			}

			/// <summary>
			/// Call after_kernel_launch() to check for any errors in launching the kernel.
			/// This will check for all pending errors.
			/// </summary>
			void	after_kernel_launch_aux(const char* pszSourceFile, int nLine)
			{
				// Check for launch errors.  This works best if cudaGetLastError()
				// cleared the host thread's last error variable just before launching
				// the kernel.  This is accomplished by the StartAsync() call.
				::wb::cuda::Throwable(cudaPeekAtLastError(), pszSourceFile, nLine);
			}

			#define before_kernel_launch()		before_kernel_launch_aux(__FILE__, __LINE__)		
			#define after_kernel_launch()		after_kernel_launch_aux(__FILE__, __LINE__)		
			#endif

			#pragma endregion			

			#pragma region "Direct Element Access"

			/// <summary>
			/// Retrieves a pointer to a particular element in the ndarray.  The pointer can point to either
			/// host or device memory.
			/// </summary>
			ElementType* get_element_pointer(int... position) {				
				memory::NDArrayBufferView& view = get_writable_view();

				// Dot product to find the byte position of the requested element.
				size_t byte_pos = view.offset;
				if (sizeof...(position) != view.strides.size())
					throw ArgumentOutOfRangeException("Call to get_element_pointer() must have the same number of arguments as dimensions of the ndarray.");
				for (auto ii = 0; ii < sizeof...(position); ii++) byte_pos += (position[ii] * view.strides[ii]);

				return *(ElementType*)((byte*)view.p_buffer->get_pointer() + byte_pos);
			}

			ElementType& operator() (int... position) {
				to_host();
				memory::NDArrayBufferView& view = get_writable_view();

				// Dot product to find the byte position of the requested element.
				size_t byte_pos = view.offset;
				if (sizeof...(position) != view.strides.size())
					throw ArgumentOutOfRangeException("Call to operator() must have the same number of arguments as dimensions of the ndarray.");
				for (auto ii = 0; ii < sizeof...(position); ii++) byte_pos += (position[ii] * view.strides[ii]);

				return *(ElementType*)((byte*)dynamic_cast<HostBuffer*>(view.p_buffer.get())->get_host_pointer() + byte_pos);
			}

			const ElementType& operator() (int... position) const {
				memory::NDArrayBufferView& view = get_readable_view_index(buffer_constraint::host);

				// Dot product to find the byte position of the requested element.
				size_t byte_pos = view.offset;
				if (sizeof...(position) != view.strides.size())
					throw ArgumentOutOfRangeException("Call to operator() must have the same number of arguments as dimensions of the ndarray.");
				for (auto ii = 0; ii < sizeof...(position); ii++) byte_pos += (position[ii] * view.strides[ii]);

				return *(const ElementType*)((byte*)dynamic_cast<HostBuffer*>(view.p_buffer.get())->get_host_pointer() + byte_pos);
			}

			#pragma endregion
		};

} }	// end namespaces

#endif	// __WB_ndarray_h__

//	End of ndarray.h
