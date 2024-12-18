/////////
//	ndarray_helpers.h
/////////

#ifndef __WB_ndarray_helpers_h__
#define __WB_ndarray_helpers_h__

/** Dependencies **/

#include "../../wbFoundation.h"
#include "../../System/GPU.h"

namespace wb {
	namespace math {

		template<typename T>
		inline T dot_product(const vector<T>& a, const vector<T>& b)
		{
			if (a.size() != b.size())
				throw ArgumentOutOfRangeException("Dot product can only be calculated between two vectors of the same length.");
			T result = (T)0;
			for (size_t ii = 0; ii < a.size(); ii++)
			{
				result += a[ii] * b[ii];
			}
			return result;
		}

		namespace memory {

			#pragma region "Memory layout helpers"

			/// <summary>
			/// The calculate_strides() function is used to calculate strides given constraints on
			/// alignment, array shape, and the basic element size.
			/// 
			/// The alignments vector specifies the alignments relative to the last element.  For example,
			/// an alignments vector of {16, 8, 2} is specifying that at least 2 bytes are required for
			/// each element and that if a 1-byte element is provided then a 1 byte gap should be appended
			/// after each element until each element is aligned to 2 bytes.  In addition, this example
			/// specifies that each row must be aligned on 8-byte boundaries.  If the shape specifies
			/// only 1D (e.g. {6}), then only the per-element entry of alignments is relevant.  If the
			/// shape specifies 2D (e.g. {4, 6}) then the last two entries of alignments is relevant.
			/// If the shape specifies more dimensions than alignments covers the higher dimensions
			/// are assumed to have no further constraints.
			/// 
			/// Note that calculate_strides() always sets up row-major order arrays, however views can
			/// be established that do not use row-major order (such views are not established through
			/// calculate_strides() but might come from transpose() or a user-specified strides vector).
			/// </summary>
			/// <returns>The strides required, in bytes, to meet the alignment, shape, and element_size constraints.</returns>
			inline vector<size_t> calculate_strides(const vector<size_t>& alignments, const vector<size_t>& shape, size_t element_size)
			{
				vector<size_t> strides;

				// example:		shape = {x, 2} with element_size being 3 (RGB).
				//	If alignment = {1} then the strides should be {2*3, 3}.
				//	If alignment = {8} then the strides should be {2*8, 8}.
				// example:		shape = {x, 2} with element_size being 16 bytes.
				//	If alignment = {3} then the strides should be {2*18, 18}.	
				// example:		shape = {x, 2} with element_size being 4 (float).
				//	If alignment = {} then the strides should be {2*4, 4}.
				//	If alignment = {8} then the strides should be {2*8, 8}.
				//	If alignment = {64, 8} then the strides should be {64, 8}.
				// example:		shape = {x, 9} with element_size being 4 (float).
				//	If alignment = {8} then the strides should be {9*8, 8}.
				//	If alignment = {32, 2} then the strides should be {64, 4}.
				//	If alignment = {x, 128, 1} then the strides should be {128, 4}.
				// Where 'x' represents any "don't care" value that is present but whose value
				// doesn't affect the outcome in the particular example.
				size_t i_alignment = alignments.size() - 1;
				for (size_t i_shape = shape.size(); i_shape > 0; i_shape--, i_alignment--)
				{
					size_t stride;
					if (i_shape == shape.size())
						stride = element_size;
					else
					{
						if (shape[i_shape] == 0) throw ArgumentException("Zero elements of shape are not accepted.");
						stride = shape[i_shape] * strides[strides.size() - 1];
					}
					if (i_alignment < alignments.size())
					{
						size_t past_aligned_boundary = (stride % alignments[i_alignment]);
						if (past_aligned_boundary != 0)
						{
							size_t padding = alignments[i_alignment] - past_aligned_boundary;
							stride += padding;
						}
					}
					strides.insert(strides.begin(), stride);
				}
				return strides;

				/** TODO: manual test suite for this function should become a unit test,
				*	matching the example cases listed above.
					size_t x = 9;
					cout << to_string(calculate_strides({ 1 }, { x, 2 }, 3)) << endl;
					cout << to_string(calculate_strides({ 8 }, { x, 2 }, 3)) << endl;
					cout << endl;
					cout << to_string(calculate_strides({ 3 }, { x, 2 }, 16)) << endl;
					cout << endl;
					cout << to_string(calculate_strides({}, { x, 2 }, 4)) << endl;
					cout << to_string(calculate_strides({ 8 }, { x, 2 }, 4)) << endl;
					cout << to_string(calculate_strides({ 64, 8 }, { x, 2 }, 4)) << endl;
					cout << endl;
					cout << to_string(calculate_strides({ 8 }, { x, 9 }, 4)) << endl;
					cout << to_string(calculate_strides({ 32, 2 }, { x, 9 }, 4)) << endl;
					cout << to_string(calculate_strides({ x, 128, 1 }, { x, 9 }, 4)) << endl;
				 */
			}

			inline size_t calculate_required_size(const vector<size_t>& strides, const vector<size_t>& shape)
			{
				// Regardless of the alignment vector and its requirements, the strides will have the same
				// number of elements as shape.  For example, if the shape is {3, x} and strides are {16, x},
				// then the strides indicates that there are 16 bytes per row and 3 rows, thus the total size
				// required is 3*16.  Only the first element of shape and strides are relevant to this calculation.
				return strides[0] * shape[0];
			}

			#pragma endregion						

			#pragma region "Memory movement helpers"

			enum class copy_kind
			{
				#ifdef CUDA_Support
				host_to_host = cudaMemcpyHostToHost,
				host_to_device = cudaMemcpyHostToDevice,
				device_to_host = cudaMemcpyDeviceToHost,
				device_to_device = cudaMemcpyDeviceToDevice
				#else
				host_to_host,
				host_to_device,
				device_to_host,
				device_to_device
				#endif
			};

			inline void move_memory_with_same_layout_async(
				byte* p_dst, byte* p_src, size_t n_bytes, copy_kind kind, 
				cuda::GPUStream stream = cuda::GPUStream::None()
			)
			{
				#ifdef CUDA_Support
				cudaThrowable(cudaMemcpyAsync(p_dst, p_src, n_bytes, (cudaMemcpyKind)kind, (cudaStream_t)stream));
				#else
				switch (kind)
				{
				case copy_kind::host_to_host:
					memmove(p_dst, p_src, n_bytes);
					return;
				default:
					throw NotSupportedException("CUDA Support is disabled so device memory is inaccessible for copy.");
				}
				#endif
			}			

			/// <summary>
			/// move_memory_async() initiates a move of memory between two non-overlapping memory buffers.  The strides 
			/// can be specified independently for source and destination.  The two buffers must represent N-D arrays of the
			/// same shape even though they can have different strides.  If the transfer involves device memory, then it
			/// is launched asynchronously, otherwise it is completed before return.
			/// </summary>			
			/// <param name="dst_strides">Layout strides associated with the destination buffer, in bytes per dimension.  Must 
			/// have the same dimensionality as shape with the last entry in strides representing the number of bytes between
			/// individual elements of the array (which can differ from element_size).</param>
			/// <param name="src_strides">Layout strides associated with the destination buffer, in bytes per dimension.  Must 
			/// have the same dimensionality as shape with the last entry in strides representing the number of bytes between
			/// individual elements of the array (which can differ from element_size).</param>
			/// <param name="shape">The shape of the N-D arrays represented by the memory buffers, specified in number of
			/// elements per each dimension.</param>
			/// <param name="element_size">The number of bytes comprising each individual element of the array.</param>
			inline void move_memory_async(
				byte* p_dst0, vector<size_t>& dst_strides, 
				byte* p_src0, vector<size_t>& src_strides, 
				vector<size_t>& shape, size_t element_size, 
				copy_kind kind, cuda::GPUStream stream = cuda::GPUStream::None()
			)
			{
				// TODO: needs unit testing over all the different scenarios.  Some cases to include:
				//	1. Column-major order of zero, one, or both buffers.
				//	2. Usually higher dimensions have higher memory addresses.  Much like column-major order can specify a
				//		buffer's stride as { 1, 3 }, a buffer could also use a stride such as { 1, 6*4, 4 }.  This would be
				//		one way to specify an RGBA image (single byte elements) of shape { 4, x, 6 }.  Following the dot 
				//		product rule, some addresses that could be accessed would include:
				//			stride dot position({ 0, 0, 0 }) = 0		"first pixel, R channel"
				//			stride dot position({ 1, 0, 0 }) = 1		"first pixel, G channel"
				//			stride dot position({ 0, 0, 1 }) = 4		"second pixel, R channel"
				//			stride dot position({ 0, 1, 0 }) = 24		"first pixel on second scanline, R channel"
				
				// TODO optimization: in some cases, would it benefit to setup DMA transfers?  This might already be done by
				// CUDA but perhaps even host-to-host copies could benefit in certain cases?

				size_t n_dims = shape.size();
				size_t src_padding_per_element = src_strides[src_strides.size() - 1] - element_size;
				size_t dst_padding_per_element = dst_strides[dst_strides.size() - 1] - element_size;

				if (src_padding_per_element != dst_padding_per_element)
				{
					/** Worst-case: each element must be moved individually **/
					/** Note: one buffer using column-major order and not the other also triggers this case **/

					if (kind != copy_kind::host_to_host)
						throw NotImplementedException("No support for gaps between elements in current implementation in device memory.  A kernel should be written for this purpose.");

					size_t elements_per_row = shape[shape.size() - 1];
					vector<size_t> pos;
					for (size_t i_dim = 0; i_dim < shape.size(); i_dim++) pos.push_back(0);
					for (;;)
					{
						size_t src_offset = dot_product(src_strides, pos);
						size_t dst_offset = dot_product(dst_strides, pos);
						byte* p_src = p_src0 + src_offset;
						byte* p_dst = p_dst0 + dst_offset;
						for (size_t j_element = 0; j_element < elements_per_row; j_element++)
						{
							for (size_t k_element = 0; k_element < element_size; k_element++, p_src++, p_dst++) *p_dst = *p_src;
							p_src += src_padding_per_element;
							p_dst += dst_padding_per_element;
						}
						if (n_dims == 1) return;			// if 1-D array, done.
						pos[pos.size() - 2] ++;
						size_t i_dim = pos.size() - 2;
						for (;;)
						{
							if (pos[i_dim] >= shape[i_dim])
							{
								if (i_dim == 0) return;
								pos[i_dim] -= shape[i_dim];
								pos[i_dim - 1]++;
							}
							if (i_dim == 0) break; else i_dim--;
						}
					}
				}
				else
				{
					// Calculate the smallest block of memory that can be moved in unison.  
					// That is, at what dimension is the layout identical?  Examples
					// include cases where the element layout is the same but row stride
					// is different, where the element and row layout is the same but
					// page (image) layout is different.  Or higher-dimensional cases.
					
					// I want to break the problem down to a split: what is the highest
					// dimension at which all dimensions to the right have the same memory
					// layout?  Having identified this dimension we can treat the dimension(s)
					// to the right as being unified blocks of memory to be transferred and
					// the dimensions to the left as needing stride consideration (iteration).

					// Caveat: we can do better than this by one dimension.  Example: an RGB 
					// image (3x64x64).  Assume src_strides = {64*64, 64, 1} and 
					// dst_strides = {64*128, 128, 1}.  In this example, the "identical" stride 
					// is found in the final dimension; only elements have the same stride and
					// rows do not.  However, we could move each row by performing a 64-byte 
					// copy from source to destination.  We then need a new offset for the 
					// pointers in order to copy the following row.  Thus, we actually want 
					// to be one left of the "identical layout dimension".  I will call this
					// the "block layout" dimension where we can copy everything within 
					// it and to the right of it in a single block copy.

					size_t i_block_copy_dim = src_strides.size() - 1;
					for (;;)
					{
						if (src_strides[i_block_copy_dim] == dst_strides[i_block_copy_dim])
						{
							if (i_block_copy_dim == 0)
							{
								// Special case, we can copy the entire buffer in one block.
								move_memory_with_same_layout_async(p_dst0, p_src0, src_strides[0] * shape[0], kind, stream);
								return;
							}
							i_block_copy_dim--;
						}
						else break;
					}

					// Assumption: the cudaMemcpy2DAsync() and cudaMemcpy3DAsync() are better optimized.  The
					// cudaMemcpy2D() function requires that source and destination pitches, width (in bytes),
					// and height be specified.  This implies:					
					//		a) the source and destination have the same gaps between elements (this condition
					//			has already been checked and handled by the "worst-case" clause above).
					//		b) pitch can differ between source and destination, which supports different # of
					//			bytes in each row between the source and destination.
					//		c) the highest dimension to be transferred is height.  This could be extended to
					//			higher dimensions by iterating.  If there is no padding in the higher dimensions 
					//			other than that found in rows, cudaMemcpy2D() can also copy more than 2D by 
					//			multiplying height by the product of dimensions counting rows and above (that 
					//			is, for ndarray shape of {3, 5, 64, 64} a height of 3*5*64 would work).
					//		d) the highest dimension to accept a stride is rows.  As per part (c), iteration over
					//			cudaMemcpy2D() would be required if padding is required for dimensions higher
					//			than rows.
					// 
					// Given the above constraints, it seems that it would be more efficient to use a single
					// block memory movement if the 2D layout is identical for the source and destination.  Only
					// when the row stride (pitch) differs between source and destination would there be an
					// advantage to cudaMemcpy2D().
					// 
					// TODO optimization: the cudaMemcpy3DAsync() might have advantages.  I'd need to check
					// what constraints it has.  At the least, it seems to require allocation of a 
					// cudaArray_t, which might be incompatible with other aspects of ndarray.

					#ifdef CUDA_Support
					if (n_dims - i_block_copy_dim == 2)
					{
						// e.g. src_strides = { 2*3*64*64,	3*64*64,	64*64,	64,		1 } 
						//		dst_strides = { 16*3*64*128, 3*64*128,	64*128, 128,	1 }
						//		n_dims = 5, i_block_copy_dim = 3, difference is 2.

						// Next, there are two sub-cases: A) higher dimensions have no gaps and we can merge
						// them all into a single "height" count or B) higher dimensions have strides indicating
						// additional padding and must be iterated.

						size_t src_block_size = src_strides[n_dims - 2];
						size_t dst_block_size = dst_strides[n_dims - 2];

						size_t width_in_bytes = src_strides[n_dims - 1] * shape[n_dims - 1];
						size_t height = shape[n_dims - 2];
						i_block_copy_dim = n_dims - 2;
						for (;;)
						{
							if (i_block_copy_dim == 0)
							{
								// Special case: entire buffer can be copied at once.
								height *= shape[0];								
								cudaThrowable(cudaMemcpy2DAsync(p_dst0, dst_strides[n_dims - 2], p_src0, src_strides[n_dims - 2], width_in_bytes, height, (cudaMemcpyKind)kind, stream));
								return;
							}
							if (src_strides[i_block_copy_dim - 1] / src_block_size == dst_strides[i_block_copy_dim - 1] / dst_block_size) {
								i_block_copy_dim--;
								height = src_strides[i_block_copy_dim - 1] / src_block_size;
							}
							else break;
						}
						
						// In the above example, i_block_copy_dim is now 1 and height is 3*64.

						vector<size_t> pos;
						for (size_t i_dim = 0; i_dim < shape.size(); i_dim++) pos.push_back(0);
						for (;;)
						{
							size_t src_offset = dot_product(src_strides, pos);
							size_t dst_offset = dot_product(dst_strides, pos);
							byte* p_src = p_src0 + src_offset;
							byte* p_dst = p_dst0 + dst_offset;
							
							cudaThrowable(cudaMemcpy2DAsync(p_dst, dst_strides[n_dims - 2], p_src, src_strides[n_dims - 2], width_in_bytes, height, (cudaMemcpyKind)kind, (cudaStream_t)stream));

							pos[i_block_copy_dim] += 1;

							size_t i_dim = i_block_copy_dim;
							for (;;)
							{
								if (pos[i_dim] >= shape[i_dim])
								{
									if (i_dim == 0) return;
									pos[i_dim] -= shape[i_dim];
									pos[i_dim - 1]++;
								}
								if (i_dim == 0) break; else i_dim--;
							}
						}
					}
					#endif	

					size_t block_size = min(src_strides[i_block_copy_dim], dst_strides[i_block_copy_dim]);
					
					vector<size_t> pos;
					for (size_t i_dim = 0; i_dim < shape.size(); i_dim++) pos.push_back(0);
					for (;;)
					{
						size_t src_offset = dot_product(src_strides, pos);
						size_t dst_offset = dot_product(dst_strides, pos);
						byte* p_src = p_src0 + src_offset;
						byte* p_dst = p_dst0 + dst_offset;

						// I am also copying any padding in the row.  Usually the padding is small, such as 4 to 12 bytes,
						// and is aligned better such that it can often be faster to copy it than not.  A more perfect
						// algorithm might account for this.
						move_memory_with_same_layout_async(p_dst, p_src, block_size, kind, stream);

						// Returning to the 3x64x64 RGB image (byte elements) example with strides of {64*64, 64, 1}
						// and {64*128, 128, 1}, we have just copied a block.  A block, in this case, represents a
						// single row, and so we have moved by the number of elements in a row.  A row is one dimension
						// back from individual elements.  If our position was previously {0, 0, 0} then it should now
						// be {0, 1, 0}.  

						pos[i_block_copy_dim] += 1;

						size_t i_dim = i_block_copy_dim;
						for (;;)
						{
							if (pos[i_dim] >= shape[i_dim])
							{
								if (i_dim == 0) return;
								pos[i_dim] -= shape[i_dim];
								pos[i_dim - 1]++;
							}
							if (i_dim == 0) break; else i_dim--;
						}
					}
				}
			}

			#pragma endregion

			#pragma region "Memory fill helpers"

			/// <summary>
			/// fill_memory_async() initiates a filling of a memory buffer with a constant value.
			/// </summary>
			template<typename ElementType>
			inline void fill_memory_async(
				void* p_dst0, vector<size_t>& dst_strides, 
				vector<size_t>& shape, ElementType value, 
				bool in_host = true, cuda::GPUStream stream = cuda::GPUStream::None())
			{
				// TODO optimization: would it be faster to prepare one row of elements in scratch memory and copy it to each
				// row of the buffer?

				size_t n_dims = shape.size();				

				if (!in_host)
					throw NotImplementedException("A kernel is needed for filling a device memory buffer with a single value efficiently.");

				size_t elements_per_row = shape[n_dims - 1];
				size_t element_stride = dst_strides[n_dims - 1];
				vector<size_t> pos;
				for (size_t i_dim = 0; i_dim < shape.size(); i_dim++) pos.push_back(0);
				for (size_t i_element = 0; ; )
				{
					size_t dst_offset = dot_product(dst_strides, pos);
					ElementType* p_dst = (ElementType*)((byte*)p_dst0 + dst_offset);
					for (size_t j_element = 0; j_element < elements_per_row; j_element++)
					{
						*p_dst = value;
						p_dst = (ElementType*)((byte*)p_dst + element_stride);
					}
					if (n_dims == 1) return;			// if 1-D array, done.
					pos[pos.size() - 2]++;
					size_t i_dim = pos.size() - 2;
					for (;;)
					{
						if (pos[i_dim] >= shape[i_dim])
						{
							if (i_dim == 0) return;
							pos[i_dim] -= shape[i_dim];
							pos[i_dim - 1]++;
						}
						if (i_dim == 0) break; else i_dim--;
					}
				}
			}

			/// <summary>
			/// initial_fill_memory_async() initiates a filling of a memory buffer with a constant value under the assumption
			/// that the entire buffer belongs to the caller.  This is true immediately after allocating a buffer when only
			/// a single view is associated with the buffer and that view windows the entire buffer space.  The 
			/// initial_fill_memory_async() function attempts certain optimizations if conditions allow and reverts to the
			/// fill_memory_async() function otherwise.
			/// </summary>
			template<typename ElementType>
			inline void initial_fill_memory_async(
				byte* p_dst0, vector<size_t>& dst_strides, 
				vector<size_t>& shape, ElementType value, 
				bool in_host = true, cuda::GPUStream stream = cuda::GPUStream::None()
			)
			{
				if (sizeof(value) > 1 && value != (ElementType)0)
				{
					fill_memory_async(p_dst0, dst_strides, shape, value, in_host, stream);
					return;
				}

				// Optimization possible.
				if (in_host)
					memset(p_dst0, value, calculate_required_size(dst_strides, shape));
				#ifdef CUDA_Support
				else
					cudaMemsetAsync(p_dst0, value, calculate_required_size(dst_strides, shape), (cudaStream_t)stream);
				#else
				else throw NotSupportedException("CUDA support is disabled.");
				#endif
			}			

			#pragma endregion

		}
	}
}	// End namespaces

#endif	// __WB_ndarray_helpers_h__

//	End of ndarray_helpers.h
