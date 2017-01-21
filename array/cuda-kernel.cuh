#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

/**
@brief Simple wrapper on CUDA memory to differentiate CPU vs GPU memory
*/
template <class T>
struct cuda_ptr
{
   using value_type = T;

   explicit cuda_ptr( T* ptr = nullptr ) : ptr( ptr )
   {}

   operator T*( ) const
   {
      return ptr;
   }

   T& operator*()
   {
      ensure(0, "no possible dereferencement! This memory is on the GPU!");
   }

   cuda_ptr operator + ( size_t offset ) const
   {
      return cuda_ptr( ptr + offset );
   }

   T* ptr;
};

namespace cuda
{
   template<typename T>
   void kernel_init(cuda_ptr<T> ptr, const size_t nb_elements, const T val);

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t nb_elements, cuda_ptr<T> output);

   template<typename T>
   void kernel_copy(const T* input, const size_t nb_elements, cuda_ptr<T> output);

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t nb_elements, T* output);

   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t input_stride, cuda_ptr<T> output, const size_t output_stride, const size_t nb_elements);

   template<typename T>
   void kernel_copy(const T* input, const size_t input_stride, cuda_ptr<T> output, const size_t output_stride, const size_t nb_elements);
   
   template<typename T>
   void kernel_copy(const cuda_ptr<T> input, const size_t input_stride, T* output, const size_t output_stride, const size_t nb_elements);
   //
   // TODO for CUDA support
   //
   // hemi 2: basic, reusable: https://github.com/harrism/hemi
   //
   // generic
   // https://devblogs.nvidia.com/parallelforall/simple-portable-parallel-c-hemi-2/
   // https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
   // https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
   // https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
   //
   // range
   // https://github.com/harrism/cpp11-range/tree/70f844968c5f669ce85f8ce4cbd24a3584c57f4b
   //
   // custom allocator
   // https://en.wikipedia.org/wiki/Buddy_memory_allocation
   // http://www.brpreiss.com/books/opus4/html/page431.html
   //
   // execution policies
   // streams
   //
}

namespace details
{
   template <class T>
   void copy_naive(cuda_ptr<T> y_pointer, ui32 y_stride, const cuda_ptr<T> x_pointer, ui32 x_stride, ui32 nb_elements)
   {
      if (y_stride == 1 && x_stride == 1)
      {
         cuda::kernel_copy(x_pointer, nb_elements, y_pointer);
      }
      else {
         cuda::kernel_copy(x_pointer, x_stride, y_pointer, x_stride, nb_elements);
      }
   }

   template <class T>
   void copy_naive(T* y_pointer, ui32 y_stride, const cuda_ptr<T> x_pointer, ui32 x_stride, ui32 nb_elements)
   {
      if (y_stride == 1 && x_stride == 1)
      {
         cuda::kernel_copy(x_pointer, nb_elements, y_pointer);
      }
      else
      {
         cuda::kernel_copy(x_pointer, x_stride, y_pointer, x_stride, nb_elements);
      }
   }

   template <class T>
   void copy_naive(cuda_ptr<T> y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements)
   {
      if (y_stride == 1 && x_stride == 1)
      {
         cuda::kernel_copy(x_pointer, nb_elements, y_pointer);
      }
      else
      {
         cuda::kernel_copy(x_pointer, x_stride, y_pointer, x_stride, nb_elements);
      }
   }
}

DECLARE_NAMESPACE_NLL_END

#endif