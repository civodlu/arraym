#include "wrapper-cublas.h"


#ifdef WITH_CUDA
#pragma warning(disable:4505) // unreferenced local function

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "cublas-common.h"

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace detail
{
   /**
   @brief Takes care of the initialization.

   @Note that the first call to init() is slow (about 0.5sec) and could bias benchmarks!
   */
   class CublasConfig
   {
   public:
      CublasConfig()
      {
      }

      void init()
      {
         if (_handle != nullptr)
         {
         } else 
         {
            auto r = cublasCreate(&_handle);
            ensure(r == CUBLAS_STATUS_SUCCESS, "CUBLAS init failed!");
         }
      }

      ~CublasConfig()
      {
         if (_handle != nullptr)
         {
            auto r = cublasDestroy(_handle);
            ensure((r == CUBLAS_STATUS_SUCCESS), "cublas shut down failed!");
         }
      }

      cublasHandle_t handle() const
      {
         return _handle;
      }

   private:
      cublasHandle_t _handle = nullptr;
   };

   CublasConfig config;


   template <class T>
   void free_gpu(T* gpu_ptr)
   {
      if (gpu_ptr != nullptr)
      {
         CHECK_CUDA(cudaFree(gpu_ptr));
      }
   }

   template <class T>
   using gpu_ptr_type = std::unique_ptr<T, void(*)(T*)>;

   template <class T>
   gpu_ptr_type<T> allocate_gpu(int nb_elements)
   {
      T* gpu_ptr = nullptr;
      CHECK_CUDA(cudaMalloc(&gpu_ptr, nb_elements * sizeof(T)));
      return gpu_ptr_type<T>(gpu_ptr, &free_gpu<T>);
   }

   template <class T>
   gpu_ptr_type<T> matrix_cpu_to_gpu(int rows, int cols, const T* A, int lda)
   {
      auto ptr = allocate_gpu<T>(rows * cols);
      CHECK_CUDA(cublasSetMatrix(rows, cols, sizeof(T), A, lda, ptr.get(), rows));
      return std::move(ptr);
   }

   template <class T>
   gpu_ptr_type<T> vector_cpu_to_gpu(int n, const T* A, int stride_a)
   {
      auto ptr = allocate_gpu<T>(n);
      CHECK_CUDA(cublasSetVector(n, sizeof(T), A, stride_a, ptr.get(), 1));
      return std::move(ptr);
   }

   template <class T>
   void matrix_gpu_to_cpu(int rows, int cols, const T* gpu_ptr, int ldgpu, T* A, int lda)
   {
      CHECK_CUDA(cublasGetMatrix(rows, cols, sizeof(T), gpu_ptr, ldgpu, A, lda));
   }

   template <class T>
   void vector_gpu_to_cpu(int n, const T* gpu_ptr, int stride_gpu, T* A, int stride_a)
   {
      CHECK_CUDA(cublasGetVector(n, sizeof(T), gpu_ptr, stride_gpu, A, stride_a));
   }

   BlasInt sgemm_cublas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
      const BlasReal beta, BlasReal* C, const BlasInt ldc)
   {
      if (matrixOrder == CBLAS_ORDER::CblasRowMajor)
      {
         // TODO should be handle different memory orders? For now, no
         return -1;
      }

      // copy the CPU based matrices to GPU
      auto gpu_ptr_a = matrix_cpu_to_gpu(M, K, A, lda);
      auto gpu_ptr_b = matrix_cpu_to_gpu(K, N, B, ldb);
      auto gpu_ptr_c = allocate_gpu<float>(M*N);

      // run calculation on the GPU
      auto transa = is_transposed(matrixOrder, TransA);
      auto transb = is_transposed(matrixOrder, TransB);

      const int ld_gpu = M;

      INIT_AND_CHECK_CUDA(cublasSgemm(config.handle(), transa, transb, M, N, K, &alpha, gpu_ptr_a.get(), lda, gpu_ptr_b.get(), ldb, &beta, gpu_ptr_c.get(), ld_gpu));

      // get the data back on CPU
      matrix_gpu_to_cpu(M, N, gpu_ptr_c.get(), ld_gpu, C, ldc);
      return 0;
   }

   BlasInt dgemm_cublas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
      const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc)
   {
      if ( matrixOrder == CBLAS_ORDER::CblasRowMajor )
      {
         // TODO should be handle different memory orders? For now, no
         return -1;
      }

      // copy the CPU based matrices to GPU
      auto gpu_ptr_a = matrix_cpu_to_gpu( M, K, A, lda );
      auto gpu_ptr_b = matrix_cpu_to_gpu( K, N, B, ldb );
      auto gpu_ptr_c = allocate_gpu<double>( M*N );

      // run calculation on the GPU
      auto transa = is_transposed( matrixOrder, TransA );
      auto transb = is_transposed( matrixOrder, TransB );

      const int ld_gpu = M;

      INIT_AND_CHECK_CUDA( cublasDgemm( config.handle(), transa, transb, M, N, K, &alpha, gpu_ptr_a.get(), lda, gpu_ptr_b.get(), ldb, &beta, gpu_ptr_c.get(), ld_gpu ) );

      // get the data back on CPU
      matrix_gpu_to_cpu( M, N, gpu_ptr_c.get(), ld_gpu, C, ldc );
      return 0;
   }

   BlasInt saxpy_cublas(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy)
   {
      auto gpu_ptr_x = vector_cpu_to_gpu(N, x, incx);
      auto gpu_ptr_y = vector_cpu_to_gpu(N, y, incy);

      INIT_AND_CHECK_CUDA(cublasSaxpy(config.handle(), N, &alpha, gpu_ptr_x.get(), 1,
                                                                  gpu_ptr_y.get(), 1));

      vector_gpu_to_cpu(N, gpu_ptr_y.get(), 1, y, incy);
      return 0;
   }

   BlasInt daxpy_cublas(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy)
   {
      auto gpu_ptr_x = vector_cpu_to_gpu( N, x, incx );
      auto gpu_ptr_y = vector_cpu_to_gpu( N, y, incy );

      INIT_AND_CHECK_CUDA( cublasDaxpy( config.handle(), N, &alpha, gpu_ptr_x.get(), 1,
         gpu_ptr_y.get(), 1 ) );

      vector_gpu_to_cpu( N, gpu_ptr_y.get(), 1, y, incy );
      return 0;
   }
}
}

DECLARE_NAMESPACE_NLL_END

#endif