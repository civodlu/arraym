#include "cuda-utils.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include "memory-gpu-cuda.cuh"


DECLARE_NAMESPACE_NLL

namespace cuda
{
   template<typename T>
   __global__ void _init_kernel(T* ptr, const T val, const size_t nb_elements)
   {
      int tidx = threadIdx.x + blockDim.x * blockIdx.x;
      const int stride = blockDim.x * gridDim.x;
      for (; tidx < nb_elements; tidx += stride)
      {
         ptr[tidx] = val;
      }
   }

   template<typename T>
   void init_kernel(T* ptr, const T val, const size_t nb_elements)
   {
      /*
      int blockSize;   // The launch configurator returned block size 
      int minGridSize; // The minimum grid size needed to achieve the 
      // maximum occupancy for a full device launch 
      int gridSize;    // The actual grid size needed, based on input size 

      cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
         &_init_kernel<T>, 0, 0);
         */
      // TODO see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
      // cudaOccupancyMaxPotentialBlockSize
      const size_t block_size = 32;
      _init_kernel<T> <<<nb_elements / block_size, block_size>>>(ptr, val, nb_elements);
      cudaDeviceSynchronize();
      cudaCheck();
   }

   template ARRAY_API void init_kernel(float* ptr, const float val, const size_t nb_elements);
}

DECLARE_NAMESPACE_NLL_END

#endif