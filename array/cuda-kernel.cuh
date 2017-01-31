#pragma once

DECLARE_NAMESPACE_NLL

#ifdef WITH_CUDA

/**
define this flag to enable the dereferencement of a single ptr. This should be avoided for performance reason
But this can be useful when performance is not an issue (e.g., to test the cuda based array, we can reuse all
the tests of the CPU arrays)

Note: using a static variable to store the actual value. So invalid in multithreading.
*/
//#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT

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
#ifdef ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
      // this is mostly for test purposes so that we can have a drop in replacement of the CPU based array
      // we need several "banks" as we can do array1[0] + array2[0] overriding the static variable
      static T values[1000];
      static int index = 0;

      index = (index + 1) % 1000;
      cudaAssert(cudaMemcpy(values+index, ptr, sizeof(T), cudaMemcpyKind::cudaMemcpyDeviceToHost));
      return *(values + index);
#else
      ensure(0, "no possible dereferencement! This memory is on the GPU!");
#endif
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
         cuda::kernel_copy(x_pointer, x_stride, y_pointer, y_stride, nb_elements);
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
         cuda::kernel_copy(x_pointer, x_stride, y_pointer, y_stride, nb_elements);
      }
   }

   template <class T>
   void cos(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sin(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sqrt(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sqr(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void abs(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void exp(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void log(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);
}

#endif // WITH_CUDA

template <class T, class NewType>
struct RebindPointer;

template <class T, class NewType>
struct RebindPointer<T*, NewType>
{
   using type = NewType*;
};

#ifdef WITH_CUDA
template <class T, class NewType>
struct RebindPointer<cuda_ptr<T>, NewType>
{
   using type = cuda_ptr<NewType>;
};
#endif

DECLARE_NAMESPACE_NLL_END
