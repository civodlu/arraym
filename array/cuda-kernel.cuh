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

   //
   // TODO: hemi 2 https://github.com/harrism/hemi
   // https://devblogs.nvidia.com/parallelforall/simple-portable-parallel-c-hemi-2/
   // https://devblogs.nvidia.com/parallelforall/how-access-global-memory-efficiently-cuda-c-kernels/
   // https://devblogs.nvidia.com/parallelforall/how-optimize-data-transfers-cuda-cc/
   // https://devblogs.nvidia.com/parallelforall/how-overlap-data-transfers-cuda-cc/
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
         ensure(0, "TODO implement!");
      }
   }
}

template <class T>
void memcpy(cuda_ptr<T> destination, const cuda_ptr<T> source, size_t size_bytes)
{
   NLL_FAST_ASSERT(size_bytes % sizeof(T) == 0, "error! Must be no rounding!");
   cuda::kernel_copy(source, size_bytes / sizeof(T), destination);
}

inline void memcpy(void* destination, const void* source, size_t size_bytes)
{
   std::memcpy(destination, source, size_bytes);
}

DECLARE_NAMESPACE_NLL_END

#endif