#pragma once

#include <array/config.h>
#include <array/array-api.h>

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace cuda
{
ARRAY_API void _cudaAssert(int code, const char* file, int line);
ARRAY_API void _cudaCheck(const char* file, int line);
}

/// assert there was no CUDA error
#define cudaAssert(ans) cuda::_cudaAssert((ans), __FILE__, __LINE__)

/// assert there was no error during kernel execution
#define cudaCheck() cuda::_cudaCheck(__FILE__, __LINE__)

namespace cuda
{
template <class T>
void free_gpu(T* gpu_ptr)
{
   if (gpu_ptr != nullptr)
   {
      cudaAssert(cudaFree(gpu_ptr));
   }
}

template <class T>
using gpu_ptr_type = std::unique_ptr<T, void (*)(T*)>;

template <class T>
gpu_ptr_type<T> allocate_gpu(int nb_elements)
{
   T* gpu_ptr = nullptr;
   cudaAssert(cudaMalloc(&gpu_ptr, nb_elements * sizeof(T)));
   return gpu_ptr_type<T>(gpu_ptr, &free_gpu<T>);
}

template <class T>
gpu_ptr_type<T> matrix_cpu_to_gpu(int rows, int cols, const T* A, int lda)
{
   auto ptr = allocate_gpu<T>(rows * cols);
   cudaAssert(cublasSetMatrix(rows, cols, sizeof(T), A, lda, ptr.get(), rows));
   return std::move(ptr);
}

template <class T>
gpu_ptr_type<T> vector_cpu_to_gpu(int n, const T* A, int stride_a)
{
   auto ptr = allocate_gpu<T>(n);
   cudaAssert(cublasSetVector(n, sizeof(T), A, stride_a, ptr.get(), 1));
   return std::move(ptr);
}

template <class T>
void matrix_gpu_to_cpu(int rows, int cols, const T* gpu_ptr, int ldgpu, T* A, int lda)
{
   cudaAssert(cublasGetMatrix(rows, cols, sizeof(T), gpu_ptr, ldgpu, A, lda));
}

template <class T>
void vector_gpu_to_cpu(int n, const T* gpu_ptr, int stride_gpu, T* A, int stride_a)
{
   cudaAssert(cublasGetVector(n, sizeof(T), gpu_ptr, stride_gpu, A, stride_a));
}
}

DECLARE_NAMESPACE_NLL_END

#endif
