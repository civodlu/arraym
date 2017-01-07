#pragma once

#include <array/array-api.h>
#include <array/config.h>
#include "wrapper-common.h"

#ifdef WITH_OPENBLAS

#pragma warning(disable : 4099) // debug info not found in dependencies

/**
 @brief This file presents the wrapers of the unoptimized BLAS library f2c provided with CLAPACK

 Dependencies are hidded in the cpp
 */

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace detail
{
//
// The CBLAS wrapper doesn't usually return BlasInt. This is added, so that if the function can handle the operation, it returns
// 0, anything else otherwise. This will enable chaining multiple implementation (i.e., one implementation doesn't have to provide
// all possible paths)
//

// CBLAS Bindings provided by OpenBLAS
ARRAY_API BlasInt saxpy_cblas(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy);
ARRAY_API BlasInt daxpy_cblas(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy);

ARRAY_API BlasInt sgemm_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                           const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
                           const BlasReal beta, BlasReal* C, const BlasInt ldc);
ARRAY_API BlasInt dgemm_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                           const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
                           const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc);

ARRAY_API BlasInt sscal_cblas(const BlasInt N, const BlasReal alpha, BlasReal* X, const BlasInt incX);
ARRAY_API BlasInt dscal_cblas(const BlasInt N, const BlasDoubleReal alpha, BlasDoubleReal* X, const BlasInt incX);

ARRAY_API BlasReal sdot_cblas(const BlasInt N, const BlasReal* x, const BlasInt incX, const BlasReal* y, const BlasInt incY);
ARRAY_API BlasDoubleReal ddot_cblas(const BlasInt N, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY);

ARRAY_API BlasReal snrm2_cblas(const BlasInt N, const BlasReal* X, const BlasInt incX);
ARRAY_API BlasDoubleReal dnrm2_cblas(const BlasInt N, const BlasDoubleReal* X, const BlasInt incX);

ARRAY_API BlasReal sasum_cblas(const BlasInt n, const BlasReal* x, const BlasInt incx);
ARRAY_API BlasDoubleReal dasum_cblas(const BlasInt n, const BlasDoubleReal* x, const BlasInt incx);

ARRAY_API BlasInt sger_cblas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* x, const BlasInt incX,
                          const BlasReal* y, const BlasInt incY, BlasReal* A, const BlasInt lda);
ARRAY_API BlasInt dger_cblas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha, const BlasDoubleReal* x, const BlasInt incX,
                          const BlasDoubleReal* y, const BlasInt incY, BlasDoubleReal* A, const BlasInt lda);

ARRAY_API BlasInt sgemv_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasReal alpha,
                           const BlasReal* A, const BlasInt lda, const BlasReal* x, const BlasInt incX, const BlasReal beta, BlasReal* y, const BlasInt incY);
ARRAY_API BlasInt dgemv_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha,
                           const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal beta,
                           BlasDoubleReal* y, const BlasInt incY);

// LAPACKE bindings
ARRAY_API BlasInt sgetrf_cblas(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasInt* ipiv);
ARRAY_API BlasInt dgetrf_cblas(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt* ipiv);
ARRAY_API BlasInt sgetri_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, const BlasInt* ipiv);
ARRAY_API BlasInt dgetri_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv);

ARRAY_API BlasInt sgesdd_cblas(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasReal* s, BlasReal* u, BlasInt ldu,
                               BlasReal* vt, BlasInt ldvt);

ARRAY_API BlasInt dgesdd_cblas(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* s, BlasDoubleReal* u,
                               BlasInt ldu, BlasDoubleReal* vt, BlasInt ldvt);

ARRAY_API BlasInt slaswp_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx);

ARRAY_API BlasInt dlaswp_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx);

ARRAY_API BlasInt sgetrs_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasReal* a, BlasInt lda, const BlasInt* ipiv, BlasReal* b,
                               BlasInt ldb);

ARRAY_API BlasInt dgetrs_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv,
                               BlasDoubleReal* b, BlasInt ldb);

ARRAY_API BlasInt sgels_cblas(CBLAS_ORDER matrix_layout, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasReal* a, BlasInt lda, BlasReal* b, BlasInt ldb);

ARRAY_API BlasInt dgels_cblas(CBLAS_ORDER matrix_layout, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* b,
                              BlasInt ldb);
}
}

DECLARE_NAMESPACE_NLL_END

#endif