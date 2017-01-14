#include "cuda-utils.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda-kernel.cuh"
#include <thrust/device_vector.h>
#include <thrust/fill.h>

DECLARE_NAMESPACE_NLL

namespace cuda
{
   template<typename T>
   __global__ void _kernel_init(T* ptr, const size_t nb_elements, const T val)
   {
      int tidx = threadIdx.x + blockDim.x * blockIdx.x;
      const int stride = blockDim.x * gridDim.x;
      for (; tidx < nb_elements; tidx += stride)
      {
         ptr[tidx] = val;
      }

      //
      // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize,
      // &_kernel_init<T>, 0, 0);
      //
   }

   /*
   template<typename T>
   void kernel_init(T* ptr, const T val, const size_t nb_elements)
   {      
      // TODO see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
      // cudaOccupancyMaxPotentialBlockSize
      const size_t block_size = 256;
      _kernel_init<T> <<<(nb_elements + block_size - 1 ) / block_size, block_size >>>(ptr, val, nb_elements);
      cudaDeviceSynchronize();
      cudaCheck();
      
      //thrust::device_ptr<T> dev_ptr(ptr);
      //thrust::fill(dev_ptr, dev_ptr + nb_elements, val);
   }*/

   template<typename T>
   void kernel_init(T* ptr, const size_t nb_elements, const T val)
   {
      thrust::device_ptr<T> dev_ptr(ptr);
      thrust::fill(dev_ptr, dev_ptr + nb_elements, val);
   }

   template<typename T>
   void kernel_copy(const T* input, const size_t nb_elements, T* output)
   {
      thrust::device_ptr<T> dev_ptr_in(const_cast<T*>(input));
      thrust::device_ptr<T> dev_ptr_out(output);
      thrust::copy(dev_ptr_in, dev_ptr_in + nb_elements, dev_ptr_out);
   }

   template ARRAY_API void kernel_init(float* ptr, const size_t nb_elements, const float val);
   template ARRAY_API void kernel_copy(const float* input, const size_t nb_elements, float* output);

}

DECLARE_NAMESPACE_NLL_END

#endif