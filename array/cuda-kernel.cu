#include "cuda-utils.h"

#ifdef WITH_CUDA

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda-kernel.cuh"
#include "cuda-utils.h"
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include "wrapper-cublas.h"

DECLARE_NAMESPACE_NLL

//
// TODO optimize all these kernels. For now they are just very simple implementations
//
// TODO see https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-occupancy-api-simplifies-launch-configuration/
// cudaOccupancyMaxPotentialBlockSize
namespace cuda
{
   /**
    @brief strided range
    see https://github.com/thrust/thrust/blob/master/examples/strided_range.cu
    */
   template <typename Iterator>
   class strided_range
   {
   public:
      typedef typename thrust::iterator_difference<Iterator>::type difference_type;

      struct stride_functor : public thrust::unary_function<difference_type, difference_type>
      {
         difference_type stride;

         stride_functor(difference_type stride)
            : stride(stride) {}

         __host__ __device__
         difference_type operator()(const difference_type& i) const
         {
            return stride * i;
         }
      };

      typedef typename thrust::counting_iterator<difference_type>                   CountingIterator;
      typedef typename thrust::transform_iterator<stride_functor, CountingIterator> TransformIterator;
      typedef typename thrust::permutation_iterator<Iterator, TransformIterator>    PermutationIterator;

      // type of the strided_range iterator
      typedef PermutationIterator iterator;

      // construct strided_range for the range [first,last)
      strided_range(Iterator first, Iterator last, difference_type stride)
         : first(first), last(last), stride(stride) {}

      iterator begin() const
      {
         return PermutationIterator(first, TransformIterator(CountingIterator(0), stride_functor(stride)));
      }

      iterator end() const
      {
         return begin() + ((last - first) + (stride - 1)) / stride;
      }

   protected:
      Iterator first;
      Iterator last;
      difference_type stride;
   };

   template<typename T>
   __global__ void _kernel_init_dummy(T* ptr, const size_t nb_elements, const T val)
   {
      int tidx = threadIdx.x + blockDim.x * blockIdx.x;
      const int stride = blockDim.x * gridDim.x;
      for (; tidx < nb_elements; tidx += stride)
      {
         ptr[tidx] = val;
      }
   }

   template<typename T>
   __global__ void _kernel_copy_dummy(const T* input, const size_t input_stride, T* output, const size_t output_stride, const size_t nb_elements)
   {
      const size_t tidx = (threadIdx.x + blockDim.x * blockIdx.x);
      const size_t tidx_input = tidx * input_stride;
      const size_t tidx_output = tidx * output_stride;
      output[tidx_output] = input[tidx_input];
   }

   
   template<typename T>
   void kernel_init_dummy(T* ptr, const T val, const size_t nb_elements)
   {      
      const size_t block_size = 256;
      _kernel_init_dummy<T> << <(nb_elements + block_size - 1) / block_size, block_size >> >(ptr, val, nb_elements);
      cudaDeviceSynchronize();
      cudaCheck();
   }

   template<typename T>
   void kernel_init(cuda_ptr<T> ptr, const size_t nb_elements, const T val)
   {
      thrust::device_ptr<T> dev_ptr((T*)ptr);
      thrust::fill(dev_ptr, dev_ptr + nb_elements, val);
   }

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t input_stride, cuda_ptr<T> output, const size_t output_stride, const size_t nb_elements)
   {
      cudaAssert(cudaMemcpy2D(
         output,
         sizeof(T) * output_stride,
         input,
         sizeof(T) * input_stride,
         sizeof(T),
         nb_elements,
         cudaMemcpyKind::cudaMemcpyDeviceToDevice));
   }

   template<typename T>
   void kernel_copy(const T* input, const size_t input_stride, cuda_ptr<T> output, const size_t output_stride, const size_t nb_elements)
   {
      cudaAssert(cudaMemcpy2D(
         output,
         sizeof(T) * output_stride,
         input,
         sizeof(T) * input_stride,
         sizeof(T),
         nb_elements,
         cudaMemcpyKind::cudaMemcpyHostToDevice));
   }

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t input_stride, T* output, const size_t output_stride, const size_t nb_elements)
   {
      cudaAssert(cudaMemcpy2D(
         output,
         sizeof(T) * output_stride,
         input,
         sizeof(T) * input_stride,
         sizeof(T),
         nb_elements,
         cudaMemcpyKind::cudaMemcpyDeviceToHost));
   }

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t nb_elements, cuda_ptr<T> output)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)(input));
      thrust::device_ptr<T> dev_ptr_out((T*)output);
      thrust::copy(dev_ptr_in, dev_ptr_in + nb_elements, dev_ptr_out); 
   }

   template<typename T>
   void kernel_copy(const T* input, const size_t nb_elements, cuda_ptr<T> output)
   {
      thrust::device_ptr<T> dev_ptr_out((T*)output);
      thrust::copy(input, input + nb_elements, dev_ptr_out);
   }

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t nb_elements, T* output)
   {
      thrust::device_ptr<T> dev_ptr_in((T*)(input));
      thrust::copy(dev_ptr_in, dev_ptr_in + nb_elements, output);
   }

   template ARRAY_API void kernel_init(cuda_ptr<float> ptr, const size_t nb_elements, const float val);
   template ARRAY_API void kernel_copy(const cuda_ptr<float> input, const size_t nb_elements, cuda_ptr<float> output);
   template ARRAY_API void kernel_copy(const cuda_ptr<float> input, const size_t nb_elements, float* output);
   template ARRAY_API void kernel_copy(const float* input, const size_t nb_elements, cuda_ptr<float> output);
   template ARRAY_API void kernel_copy(const cuda_ptr<float> input, const size_t input_stride, cuda_ptr<float> output, const size_t output_stride, const size_t nb_elements);
   template ARRAY_API void kernel_copy(const float* input, const size_t input_stride, cuda_ptr<float> output, const size_t output_stride, const size_t nb_elements);
   template ARRAY_API void kernel_copy(const cuda_ptr<float> input, const size_t input_stride, float* output, const size_t output_stride, const size_t nb_elements);

   template ARRAY_API void kernel_init( cuda_ptr<double> ptr, const size_t nb_elements, const double val );
   template ARRAY_API void kernel_copy( const cuda_ptr<double> input, const size_t nb_elements, cuda_ptr<double> output );
   template ARRAY_API void kernel_copy( const cuda_ptr<double> input, const size_t nb_elements, double* output );
   template ARRAY_API void kernel_copy( const double* input, const size_t nb_elements, cuda_ptr<double> output );
   template ARRAY_API void kernel_copy( const cuda_ptr<double> input, const size_t input_stride, cuda_ptr<double> output, const size_t output_stride, const size_t nb_elements );
   template ARRAY_API void kernel_copy( const double* input, const size_t input_stride, cuda_ptr<double> output, const size_t output_stride, const size_t nb_elements );
   template ARRAY_API void kernel_copy( const cuda_ptr<double> input, const size_t input_stride, double* output, const size_t output_stride, const size_t nb_elements );
}

namespace details
{
   
}

DECLARE_NAMESPACE_NLL_END

#endif