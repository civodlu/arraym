#pragma once

#include <array/array-api.h>
#include <array/config.h>
#include "wrapper-common.h"

#ifdef WITH_CUDA

#include <cublas_v2.h>

/**
@brief this file defines a wrapper for cublas/cublas-XT
*/

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
      CublasConfig();
      void init();
      ~CublasConfig();
      cublasHandle_t handle() const;

   private:
      cublasHandle_t _handle = nullptr;
   };

   extern CublasConfig config;

   ARRAY_API BlasInt saxpy_cublas(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy);
   ARRAY_API BlasInt daxpy_cublas(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy);

   ARRAY_API BlasInt sgemm_cublas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
      const BlasReal beta, BlasReal* C, const BlasInt ldc);

   ARRAY_API BlasInt dgemm_cublas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
      const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc);

   ARRAY_API BlasInt sscal_cublas(const BlasInt N, const BlasReal alpha, BlasReal* X, const BlasInt incX);
   ARRAY_API BlasInt dscal_cublas(const BlasInt N, const BlasDoubleReal alpha, BlasDoubleReal* X, const BlasInt incX);

   ARRAY_API BlasReal sdot_cublas(const BlasInt N, const BlasReal* x, const BlasInt incX, const BlasReal* y, const BlasInt incY);
   ARRAY_API BlasDoubleReal ddot_cublas(const BlasInt N, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY);

   ARRAY_API BlasReal snrm2_cublas(const BlasInt N, const BlasReal* X, const BlasInt incX);
   ARRAY_API BlasDoubleReal dnrm2_cublas(const BlasInt N, const BlasDoubleReal* X, const BlasInt incX);

   ARRAY_API BlasReal sasum_cublas(const BlasInt n, const BlasReal* x, const BlasInt incx);
   ARRAY_API BlasDoubleReal dasum_cublas(const BlasInt n, const BlasDoubleReal* x, const BlasInt incx);

   ARRAY_API BlasInt sger_cublas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* x, const BlasInt incX,
      const BlasReal* y, const BlasInt incY, BlasReal* A, const BlasInt lda);
   ARRAY_API BlasInt dger_cublas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha, const BlasDoubleReal* x, const BlasInt incX,
      const BlasDoubleReal* y, const BlasInt incY, BlasDoubleReal* A, const BlasInt lda);

}
}

DECLARE_NAMESPACE_NLL_END

#endif