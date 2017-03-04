#include "wrapper-cblas.h"

#ifdef WITH_OPENBLAS

#ifdef _MSC_VER
//#pragma comment(lib, "libopenblas.lib")

// http://stackoverflow.com/questions/24853450/errors-using-lapack-c-header-in-c-with-visual-studio-2010
#include <complex>
#define lapack_complex_float std::complex<float>
#define lapack_complex_double std::complex<double>
#endif

#include <lapacke.h>
#include <cblas.h>

DECLARE_NAMESPACE_NLL

namespace blas
{
namespace detail
{
static_assert((int)CBLAS_ORDER::CblasColMajor == (int)LAPACK_COL_MAJOR, "must be the same value!");
static_assert((int)CBLAS_ORDER::CblasRowMajor == (int)LAPACK_ROW_MAJOR, "must be the same value!");

static ::CBLAS_TRANSPOSE transp(const enum CBLAS_TRANSPOSE v)
{
   switch (v)
   {
   case CBLAS_TRANSPOSE::CblasTrans:
      return ::CBLAS_TRANSPOSE::CblasTrans;
   case CBLAS_TRANSPOSE::CblasNoTrans:
      return ::CBLAS_TRANSPOSE::CblasNoTrans;
   }
   throw "error!";
}

static ::CBLAS_ORDER order(const enum CBLAS_ORDER v)
{
   switch (v)
   {
   case CBLAS_ORDER::CblasColMajor:
      return ::CBLAS_ORDER::CblasColMajor;
   case CBLAS_ORDER::CblasRowMajor:
      return ::CBLAS_ORDER::CblasRowMajor;
   }
   throw "order not handled!";
}

BlasInt saxpy_cblas(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy)
{
   ::cblas_saxpy(N, alpha, x, incx, y, incy);
   return 0;
}

BlasInt daxpy_cblas(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy)
{
   ::cblas_daxpy(N, alpha, x, incx, y, incy);
   return 0;
}

BlasInt sgemm_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                    const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb, const BlasReal beta,
                    BlasReal* C, const BlasInt ldc)
{
   cblas_sgemm(order(matrixOrder), transp(TransA), transp(TransB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
   return 0;
}

BlasInt dgemm_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                    const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* B, const BlasInt ldb,
                    const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc)
{
   cblas_dgemm(order(matrixOrder), transp(TransA), transp(TransB), M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
   return 0;
}

BlasInt sscal_cblas(const BlasInt N, const BlasReal alpha, BlasReal* X, const BlasInt incX)
{
   ::cblas_sscal(N, alpha, X, incX);
   return 0;
}

BlasInt dscal_cblas(const BlasInt N, const BlasDoubleReal alpha, BlasDoubleReal* X, const BlasInt incX)
{
   ::cblas_dscal(N, alpha, X, incX);
   return 0;
}

BlasReal sdot_cblas(const BlasInt N, const BlasReal* x, const BlasInt incX, const BlasReal* y, const BlasInt incY)
{
   return ::cblas_sdot(N, x, incX, y, incY);
}

BlasDoubleReal ddot_cblas(const BlasInt N, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY)
{
   return ::cblas_ddot(N, x, incX, y, incY);
}

BlasInt sger_cblas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* x, const BlasInt incX, const BlasReal* y,
                   const BlasInt incY, BlasReal* A, const BlasInt lda)
{
   ::cblas_sger(order(matrixOrder), M, N, alpha, x, incX, y, incY, A, lda);
   return 0;
}

BlasInt dger_cblas(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha, const BlasDoubleReal* x, const BlasInt incX,
                   const BlasDoubleReal* y, const BlasInt incY, BlasDoubleReal* A, const BlasInt lda)
{
   ::cblas_dger(order(matrixOrder), M, N, alpha, x, incX, y, incY, A, lda);
   return 0;
}

BlasInt sgemv_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* A,
                    const BlasInt lda, const BlasReal* x, const BlasInt incX, const BlasReal beta, BlasReal* y, const BlasInt incY)
{
   ::cblas_sgemv(order(matrixOrder), transp(TransA), M, N, alpha, A, lda, x, incX, beta, y, incY);
   return 0;
}

BlasInt dgemv_cblas(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha,
                    const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal beta, BlasDoubleReal* y,
                    const BlasInt incY)
{
   ::cblas_dgemv(order(matrixOrder), transp(TransA), M, N, alpha, A, lda, x, incX, beta, y, incY);
   return 0;
}

BlasInt sgetrf_cblas(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasInt* ipiv)
{
   return ::LAPACKE_sgetrf(order(matrixOrder), m, n, a, lda, ipiv);
}

BlasInt dgetrf_cblas(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt* ipiv)
{
   return ::LAPACKE_dgetrf(order(matrixOrder), m, n, a, lda, ipiv);
}

BlasInt sgetri_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, const BlasInt* ipiv)
{
   return ::LAPACKE_sgetri(order(matrixOrder), n, a, lda, ipiv);
}

BlasInt dgetri_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv)
{
   return ::LAPACKE_dgetri(order(matrixOrder), n, a, lda, ipiv);
}

BlasReal snrm2_cblas(const BlasInt N, const BlasReal* X, const BlasInt incX)
{
   return ::cblas_snrm2(N, X, incX);
}

BlasDoubleReal dnrm2_cblas(const BlasInt N, const BlasDoubleReal* X, const BlasInt incX)
{
   return ::cblas_dnrm2(N, X, incX);
}

BlasReal sasum_cblas(const BlasInt n, const BlasReal* x, const BlasInt incx)
{
   return ::cblas_sasum(n, x, incx);
}

BlasDoubleReal dasum_cblas(const BlasInt n, const BlasDoubleReal* x, const BlasInt incx)
{
   return ::cblas_dasum(n, x, incx);
}

BlasInt sgesdd_cblas(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasReal* s, BlasReal* u, BlasInt ldu, BlasReal* vt,
                     BlasInt ldvt)
{
   return ::LAPACKE_sgesdd(order(matrixOrder), jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

BlasInt dgesdd_cblas(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* s, BlasDoubleReal* u,
                     BlasInt ldu, BlasDoubleReal* vt, BlasInt ldvt)
{
   return ::LAPACKE_dgesdd(order(matrixOrder), jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

BlasInt slaswp_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)
{
   return ::LAPACKE_slaswp(order(matrixOrder), n, a, lda, k1, k2, ipiv, incx);
}

BlasInt dlaswp_cblas(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)
{
   return ::LAPACKE_dlaswp(order(matrixOrder), n, a, lda, k1, k2, ipiv, incx);
}

BlasInt sgetrs_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasReal* a, BlasInt lda, const BlasInt* ipiv, BlasReal* b,
                     BlasInt ldb)
{
   return ::LAPACKE_sgetrs(order(matrix_order), trans, n, nrhs, a, lda, ipiv, b, ldb);
}

BlasInt dgetrs_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv,
                     BlasDoubleReal* b, BlasInt ldb)
{
   return ::LAPACKE_dgetrs(order(matrix_order), trans, n, nrhs, a, lda, ipiv, b, ldb);
}

BlasInt sgels_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasReal* a, BlasInt lda, BlasReal* b, BlasInt ldb)
{
   return ::LAPACKE_sgels(order(matrix_order), trans, m, n, nrhs, a, lda, b, ldb);
}

BlasInt dgels_cblas(CBLAS_ORDER matrix_order, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* b, BlasInt ldb)
{
   return ::LAPACKE_dgels(order(matrix_order), trans, m, n, nrhs, a, lda, b, ldb);
}
}
}

DECLARE_NAMESPACE_NLL_END

#endif
