
#include "gtest/gtest.h"
#include "System/GPU.h"

#pragma comment(lib, "cuda.lib")	// Force linkage to cuda.lib to get access to driver API, i.e. cuGetErrorString() and cuCtxGetDevice().

extern bool CUDAImageProcessingTesting();

TEST(ImageProcessing, CUDA)
{
	ASSERT_TRUE(CUDAImageProcessingTesting());
}

