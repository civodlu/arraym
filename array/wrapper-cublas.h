#pragma once

#include <array/array-api.h>
#include <array/config.h>
#include "wrapper-common.h"

#ifdef WITH_CUDA

/**
@brief this file defines a wrapper for cublas/cublas-XT
*/

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace detail
{
   ARRAY_API BlasInt sgemm_cublasxt(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
      const BlasReal beta, BlasReal* C, const BlasInt ldc);

   ARRAY_API BlasInt dgemm_cublasxt(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
      const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
      const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc);
}
}

DECLARE_NAMESPACE_NLL_END

#endif