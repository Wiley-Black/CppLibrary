#ifndef __wbImages_Kernels_cuh__
#define __wbImages_Kernels_cuh__

#include <math.h>
//#include <complex.h>

#include "../../wbFoundation.h"
#include "../../System/GPU.h"			// Includes cuda.h, if supported.

#include <thrust/complex.h>

namespace wb {
    namespace cuda {

        struct img_size {
            int width, height;

            img_size() {
                width = height = 0;
            }
            img_size(int width, int height) {
                this->width = width;
                this->height = height;
            }
            bool operator==(const img_size& rhs) const {
                return (width == rhs.width && height == rhs.height);
            }
            bool operator!=(const img_size& rhs) const {
                return (width != rhs.width || height != rhs.height);
            }
        };

        struct img_format : public img_size {
            int stride;

            img_format() {
                width = height = stride = 0;
            }
            img_format(int width, int height, int stride)
            {
                this->width = width;
                this->height = height;
                this->stride = stride;
            }
            img_size size() const {
                return img_size(width, height);
            }
        };

        #ifdef __CUDACC__                

        template<typename T>
        __global__ void Kernel_Fill(
            const T* __restrict__ img,
            img_format format,
            T value)
        {
            int xx = blockIdx.x * blockDim.x + threadIdx.x;
            int yy = blockIdx.y * blockDim.y + threadIdx.y;

            if (xx < format.width && yy < format.height)
            {
                T* pScanline = (T*)((byte*)img + yy * format.stride);
                pScanline[xx] = value;
            }
        }

        // Kernel_Template() provides a macro that provides all the boilerplate for a
        // simple pixelwise image operation except the operation itself.  It will 
        // provide two pointers at the time of 'Operation': pA and pB, both of type
        // T*, the template parameter.  
        //
        // Precondition: caller must have checked that the image width and height
        // are identical (or that B is larger in both dimensions than A).  The following
        // line conveniently provides this given two img_format structures: 
        //      if (formatA.size() != formatB.size()) throw ...;
        // 
        // The strides of the two images can mismatch.  Failure to verify that the sizes 
        // match before the kernel call may result in a crash.  This choice is made for 
        // speed, since the additional if statement to verify would slow things down.
        #define Kernel_2Img_InPlaceTemplate(Kernel_Name, Operation)      \
            template<typename Ta, typename Tb> __global__ void Kernel_Name(       \
                Ta* __restrict__ imgAandOut,                         \
                img_format formatA,                                 \
                const Tb* __restrict__ imgB,                         \
                img_format formatB                                  \
                )                                                   \
            {                                                       \
                int xx = blockIdx.x * blockDim.x + threadIdx.x;     \
                int yy = blockIdx.y * blockDim.y + threadIdx.y;     \
                \
                if (xx < formatA.width && yy < formatA.height)      \
                {                                                           \
                    Ta* pA = (Ta*)((byte*)imgAandOut + yy * formatA.stride) + xx;   \
                    Tb* pB = (Tb*)((byte*)imgB + yy * formatB.stride) + xx;   \
                    { Operation; }                                          \
                }   \
            }

        Kernel_2Img_InPlaceTemplate(Kernel_AddInPlace, { *pA += *pB; });
        Kernel_2Img_InPlaceTemplate(Kernel_SubInPlace, { *pA -= *pB; });
        Kernel_2Img_InPlaceTemplate(Kernel_MulInPlace, { *pA *= *pB; });
        Kernel_2Img_InPlaceTemplate(Kernel_DivInPlace, { *pA /= *pB; });
        Kernel_2Img_InPlaceTemplate(Kernel_RealAbsolute, { *pA = abs(*pB); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexAbsolute, { *pA = thrust::abs(*pB); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexArgument, { *pA = thrust::arg(*pB); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexConjugate, { *pA = thrust::conj(*pB); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexSetReal, { *pA = Ta(*pB, pA->imag()); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexSetImag, { *pA = Ta(pA->real(), *pB); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexGetReal, { *pA = Ta(pB->real()); });
        Kernel_2Img_InPlaceTemplate(Kernel_ComplexGetImag, { *pA = Ta(pB->imag()); });
        #undef Kernel_2Img_InPlaceTemplate

        #define Kernel_2Img_NewTemplate(Kernel_Name, Operation)      \
            template<typename Ta, typename Tb, typename Tresult> __global__ void Kernel_Name(       \
                Ta* __restrict__ imgA,                              \
                img_format formatA,                                 \
                const Tb* __restrict__ imgB,                        \
                img_format formatB,                                 \
                const Tresult* __restrict__ imgResult,              \
                img_format formatResult                             \
                )                                                   \
            {                                                       \
                int xx = blockIdx.x * blockDim.x + threadIdx.x;     \
                int yy = blockIdx.y * blockDim.y + threadIdx.y;     \
                \
                if (xx < formatA.width && yy < formatA.height)      \
                {                                                           \
                    Ta* pA = (Ta*)((byte*)imgA + yy * formatA.stride) + xx;   \
                    Tb* pB = (Tb*)((byte*)imgB + yy * formatB.stride) + xx;   \
                    Tresult* pResult = (Tresult*)((byte*)imgResult + yy * formatResult.stride) + xx;   \
                    { Operation; }                                          \
                }   \
            }
        Kernel_2Img_NewTemplate(Kernel_Add, { *pResult = *pA + *pB; });
        Kernel_2Img_NewTemplate(Kernel_Sub, { *pResult = *pA - *pB; });
        Kernel_2Img_NewTemplate(Kernel_Mul, { *pResult = *pA * *pB; });
        Kernel_2Img_NewTemplate(Kernel_Div, { *pResult = *pA / *pB; });
        #undef Kernel_2Img_NewTemplate

        #define Kernel_1Img1Scalar_InPlaceTemplate(Kernel_Name, Operation)      \
            template<typename T, typename ScT> __global__ void Kernel_Name(     \
                const T* __restrict__ imgAandOut,                   \
                img_format formatA,                                 \
                const ScT B                                         \
                )                                                   \
            {                                                       \
                int xx = blockIdx.x * blockDim.x + threadIdx.x;     \
                int yy = blockIdx.y * blockDim.y + threadIdx.y;     \
                \
                if (xx < formatA.width && yy < formatA.height)      \
                {                                                           \
                    T* pA = (T*)((byte*)imgAandOut + yy * formatA.stride) + xx;   \
                    { Operation; }                                          \
                }   \
            }

        Kernel_1Img1Scalar_InPlaceTemplate(Kernel_AddScalarInPlace, { *pA += B; });
        Kernel_1Img1Scalar_InPlaceTemplate(Kernel_SubScalarInPlace, { *pA -= B; });
        Kernel_1Img1Scalar_InPlaceTemplate(Kernel_MulScalarInPlace, { *pA *= B; });
        Kernel_1Img1Scalar_InPlaceTemplate(Kernel_DivScalarInPlace, { *pA /= B; });
        #undef Kernel_1Img1Scalar_InPlaceTemplate

        #define Kernel_1Img_InPlaceTemplate(Kernel_Name, Operation)      \
            template<typename T> __global__ void Kernel_Name(       \
                const T* __restrict__ imgAandOut,                   \
                img_format formatA                                  \
                )                                                   \
            {                                                       \
                int xx = blockIdx.x * blockDim.x + threadIdx.x;     \
                int yy = blockIdx.y * blockDim.y + threadIdx.y;     \
                \
                if (xx < formatA.width && yy < formatA.height)      \
                {                                                           \
                    T* pA = (T*)((byte*)imgAandOut + yy * formatA.stride) + xx;   \
                    { Operation; }                                          \
                }   \
            }                
        Kernel_1Img_InPlaceTemplate(Kernel_RealAbsoluteInPlace, { *pA = abs(*pA); });
        Kernel_1Img_InPlaceTemplate(Kernel_ComplexAbsoluteInPlace, { *pA = thrust::abs(*pA); });
        Kernel_1Img_InPlaceTemplate(Kernel_ComplexConjugateInPlace, { *pA = thrust::conj(*pA); });
        #undef Kernel_1Img_InPlaceTemplate

        template<typename T> __global__ void Kernel_SaturateInPlace(
            T* __restrict__ imgAandOut, 
            img_format formatA,
            T MinValue, T MaxValue
            )
        {
            int xx = blockIdx.x * blockDim.x + threadIdx.x;
            int yy = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (xx < formatA.width && yy < formatA.height)
            {
                T* pA = (T*)((byte*)imgAandOut + yy * formatA.stride) + xx;
                *pA = (*pA < MinValue) ? MinValue : ((*pA > MaxValue) ? MaxValue : *pA);
            }
        }        

        template<typename Tdst, typename Tsrc> __global__ void Kernel_TypecastConvertTo(
            Tdst* __restrict__ imgDst, img_format formatDst,
            const Tsrc* __restrict__ imgSrc, img_format formatSrc
            )        
        {
            int xx = blockIdx.x * blockDim.x + threadIdx.x;
            int yy = blockIdx.y * blockDim.y + threadIdx.y;
            if (xx < formatSrc.width && yy < formatSrc.height)
            {
                Tsrc* pSrc = (Tsrc*)((byte*)imgSrc + yy * formatSrc.stride) + xx;
                Tdst* pDst = (Tdst*)((byte*)imgDst + yy * formatDst.stride) + xx;
                *pDst = (Tdst)*pSrc;
            }
        }

        #ifdef CUB_Support                

        #if __CUDA_ARCH__ < 600
        inline __device__ double atomicAdd(double* address, double val)
        {
            unsigned long long int* address_as_ull =
                (unsigned long long int*)address;
            unsigned long long int old = *address_as_ull, assumed;

            do {
                assumed = old;
                old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val +
                        __longlong_as_double(assumed)));

                // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            } while (assumed != old);

            return __longlong_as_double(old);
        }
        #endif
        
        template<typename AccumElemType, typename PixelType, int BlockDimX, int BlockDimY, int ItemsPerThread> __global__ void Kernel_ComplexSumBlockReduce(
            PixelType* pImg,
            img_format format,
            thrust::complex<AccumElemType>* pOut
            )
        {
            typedef thrust::complex<AccumElemType> AccumType;            

            int xx = blockDim.x * blockIdx.x + threadIdx.x;
            int yy = blockDim.y * blockIdx.y + threadIdx.y;

            // More sophisticated loading routine...  though I don't understand how we aren't reducing the number of blocks if we're doing this...
            #if 0
            // Profiled: 20 iterations required 11.8114 seconds, 3.66301e+06 pixels/second, or 0.0293041 GB/s.
            // Per-thread tile data
            AccumType items[ItemsPerThread];            
            //int linear_tid = (threadIdx.y * blockDim.x) + linear_tid;
            //LoadDirectStriped<BLOCK_THREADS>(linear_tid, d_in, data);
            #pragma unroll
            for (int ITEM = 0; ITEM < ItemsPerThread; ITEM++)
            {
                int ys = yy + (ITEM * BlockDimY);
                if (xx < format.width && ys < format.height)
                {
                    PixelType* pA = (PixelType*)((byte*)pImg + yy * format.stride) + xx + ITEM * BlockDimX;
                    items[ITEM] = (AccumType)*pA;                    
                }
                else items[ITEM] = (AccumType)0.0;
            }            
            #else   // Simpler but I think less optimal approach:
            // Profiled: 20 iterations required 11.7622 seconds, 3.67833e+06 pixels/second, or 0.0294266 GB/s.
            AccumType items;
            if (xx < format.width && yy < format.height) {
                PixelType* pA = (PixelType*)((byte*)pImg + yy * format.stride) + xx;
                items = (AccumType)*pA;
            }
            else
                items = (AccumType)0.0;
            #endif

            // Specialize BlockReduce type for our thread block
            typedef cub::BlockReduce<AccumType, BlockDimX, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, BlockDimY> BlockReduceT;
            __shared__ typename BlockReduceT::TempStorage temp_storage;

            AccumType aggregate = BlockReduceT(temp_storage).Sum(items);
            int BlocksPerScanline = gridDim.x / blockDim.x;
            // I'm not sure why we have to do an atomicAdd() here.  It isn't in the cub example.  If we leave it out, we get an answer that is
            // roughly half of correct, so it appears to be a merge of 2 per block.
            //if (threadIdx.x == 0 && threadIdx.y == 0) *(AccumElemType*)&pOut[blockIdx.y * BlocksPerScanline + blockIdx.x] = aggregate.real();
            //if (threadIdx.x == 0 && threadIdx.y == 0) *((AccumElemType*)&pOut[blockIdx.y * BlocksPerScanline + blockIdx.x] + 1) = aggregate.imag();
            if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd((AccumElemType*)&pOut[blockIdx.y * BlocksPerScanline + blockIdx.x], aggregate.real());
            if (threadIdx.x == 0 && threadIdx.y == 0) atomicAdd(((AccumElemType*)&pOut[blockIdx.y * BlocksPerScanline + blockIdx.x] + 1), aggregate.imag());
            // In profiling, removing the atomicAdds (and accepting an inaccurate result) had no discernable effect on throughput.
        }

        #endif  // CUB_Support

        #endif  // __CUDACC__
    }
}

#endif	// __wbImages_Kernels_cuh__

//	End of Images_Kernels.cuh
