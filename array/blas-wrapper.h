#pragma once

#include "blas-dispatcher.h"

//# define INSTRUMENT_BLAS_CALLS

#ifndef INSTRUMENT_BLAS
#define INSTRUMENT_BLAS(x) std::cout << "Called=" << x << std::endl
#endif

/**
@brief BLAS wrappers encapsulate different BLAS implementation. At runtime, given the arguments, the fastest BLAS
routine is called

The steps are:
- wrap a blas implementation in wrapper-blas-*.h/cpp, dependencies should be hidden in the cpp
- blas dispatcher handle the registration of the blas routines and runtime dispatch
- wrapper-blas.h which simply calls the correct blas version (BlasReal or BlasDoubleReal)
*/
DECLARE_NAMESPACE_NLL

namespace blas
{

template <class T>
T nrm2(const BlasInt N, const T* X, const BlasInt incX)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline BlasReal nrm2(const BlasInt N, const BlasReal* X, const BlasInt incX)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("nrm2<BlasReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::snrm2>(N, X, incX);
}

template <>
inline BlasDoubleReal nrm2(const BlasInt N, const BlasDoubleReal* X, const BlasInt incX)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("nrm2<BlasDoubleReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::dnrm2>(N, X, incX);
}

template <class T>
T asum(const BlasInt n, const T* x, const BlasInt incx)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline BlasReal asum(const BlasInt n, const BlasReal* x, const BlasInt incx)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("asum<BlasReal>");
#endif

   return BlasDispatcher::instance().call<details::BlasFunction::sasum>(n, x, incx);
}

template <>
inline BlasDoubleReal asum(const BlasInt n, const BlasDoubleReal* x, const BlasInt incx)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("asum<BlasReal>");
#endif

   return BlasDispatcher::instance().call<details::BlasFunction::dasum>(n, x, incx);
}

template <class T>
void axpy(BlasInt N, T alpha, const T* x, BlasInt incx, T* y, BlasInt incy)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline void axpy<BlasReal>(BlasInt N, BlasReal alpha, const BlasReal* x, BlasInt incx, BlasReal* y, BlasInt incy)
{
   BlasDispatcher::instance().call<details::BlasFunction::saxpy>(N, alpha, x, incx, y, incy);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("axpy<BlasReal>");
#endif
}

template <>
inline void axpy<BlasDoubleReal>(BlasInt N, BlasDoubleReal alpha, const BlasDoubleReal* x, BlasInt incx, BlasDoubleReal* y, BlasInt incy)
{
   BlasDispatcher::instance().call<details::BlasFunction::daxpy>(N, alpha, x, incx, y, incy);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("axpy<BlasDoubleReal>");
#endif
}

template <class T>
void gemm(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N, const BlasInt K,
          const T alpha, const T* A, const BlasInt lda, const T* B, const BlasInt ldb, const T beta, T* C, const BlasInt ldc)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline void gemm<BlasReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M, const BlasInt N,
                           const BlasInt K, const BlasReal alpha, const BlasReal* A, const BlasInt lda, const BlasReal* B, const BlasInt ldb,
                           const BlasReal beta, BlasReal* C, const BlasInt ldc)
{
   BlasDispatcher::instance().call<details::BlasFunction::sgemm>(matrixOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gemm<BlasReal>");
#endif
}

template <>
inline void gemm<BlasDoubleReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB, const BlasInt M,
                                 const BlasInt N, const BlasInt K, const BlasDoubleReal alpha, const BlasDoubleReal* A, const BlasInt lda,
                                 const BlasDoubleReal* B, const BlasInt ldb, const BlasDoubleReal beta, BlasDoubleReal* C, const BlasInt ldc)
{
   BlasDispatcher::instance().call<details::BlasFunction::dgemm>(matrixOrder, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gemm<BlasDoubleReal>");
#endif
}

template <class T>
void scal(const BlasInt N, const T alpha, T* X, const BlasInt incX)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline void scal<BlasReal>(const BlasInt N, const BlasReal alpha, BlasReal* X, const BlasInt incX)
{
   BlasDispatcher::instance().call<details::BlasFunction::sscal>(N, alpha, X, incX);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("scal<BlasReal>");
#endif
}

template <>
inline void scal<BlasDoubleReal>(const BlasInt N, const BlasDoubleReal alpha, BlasDoubleReal* X, const BlasInt incX)
{
   BlasDispatcher::instance().call<details::BlasFunction::dscal>(N, alpha, X, incX);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("scal<BlasDoubleReal>");
#endif
}

template <class T>
T dot(const BlasInt N, const T* x, const BlasInt incX, const T* y, const BlasInt incY)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline BlasReal dot<BlasReal>(const BlasInt N, const BlasReal* x, const BlasInt incX, const BlasReal* y, const BlasInt incY)
{
   return BlasDispatcher::instance().call<details::BlasFunction::sdot>(N, x, incX, y, incY);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("dot<BlasReal>");
#endif
}

template <>
inline BlasDoubleReal dot<BlasDoubleReal>(const BlasInt N, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY)
{
   return BlasDispatcher::instance().call<details::BlasFunction::ddot>(N, x, incX, y, incY);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("dot<BlasDoubleReal>");
#endif
}

template <class T>
void ger(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const T alpha, const T* x, const BlasInt incX, const T* y, const BlasInt incY, T* A,
         const BlasInt lda)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline void ger<BlasReal>(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasReal alpha, const BlasReal* x, const BlasInt incX,
                          const BlasReal* y, const BlasInt incY, BlasReal* A, const BlasInt lda)
{
   BlasDispatcher::instance().call<details::BlasFunction::sger>(matrixOrder, M, N, alpha, x, incX, y, incY, A, lda);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("ger<BlasReal>");
#endif
}

template <>
inline void ger<BlasDoubleReal>(CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha, const BlasDoubleReal* x,
                                const BlasInt incX, const BlasDoubleReal* y, const BlasInt incY, BlasDoubleReal* A, const BlasInt lda)
{
   BlasDispatcher::instance().call<details::BlasFunction::dger>(matrixOrder, M, N, alpha, x, incX, y, incY, A, lda);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("ger<BlasDoubleReal>");
#endif
}

template <class T>
void gemv(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const T alpha, const T* A, const BlasInt lda,
          const T* x, const BlasInt incX, const T beta, T* y, const BlasInt incY)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline void gemv<BlasReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasReal alpha,
                           const BlasReal* A, const BlasInt lda, const BlasReal* x, const BlasInt incX, const BlasReal beta, BlasReal* y, const BlasInt incY)

{
   BlasDispatcher::instance().call<details::BlasFunction::sgemv>(matrixOrder, TransA, M, N, alpha, A, lda, x, incX, beta, y, incY);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gemv<BlasReal>");
#endif
}

template <>
inline void gemv<BlasDoubleReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA, const BlasInt M, const BlasInt N, const BlasDoubleReal alpha,
                                 const BlasDoubleReal* A, const BlasInt lda, const BlasDoubleReal* x, const BlasInt incX, const BlasDoubleReal beta,
                                 BlasDoubleReal* y, const BlasInt incY)
{
   BlasDispatcher::instance().call<details::BlasFunction::dgemv>(matrixOrder, TransA, M, N, alpha, A, lda, x, incX, beta, y, incY);
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gemv<BlasDoubleReal>");
#endif
}

//
// LAPACK
//
template <class T>
int getrf(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, T* a, BlasInt lda, BlasInt* ipiv)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int getrf<BlasReal>(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasInt* ipiv)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("getrf<BlasReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::sgetrf>(matrixOrder, m, n, a, lda, ipiv);
}

template <>
inline int getrf<BlasDoubleReal>(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt* ipiv)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("getrf<BlasDoubleReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::dgetrf>(matrixOrder, m, n, a, lda, ipiv);
}

template <class T>
int getri(CBLAS_ORDER matrixOrder, BlasInt n, T* a, BlasInt lda, const BlasInt* ipiv)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int getri<BlasReal>(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, const BlasInt* ipiv)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("getri<BlasReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::sgetri>(matrixOrder, n, a, lda, ipiv);
}

template <>
inline int getri<BlasDoubleReal>(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("getri<BlasDoubleReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::dgetri>(matrixOrder, n, a, lda, ipiv);
}

template <class T>
int gesdd(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, T* a, BlasInt lda, T* s, T* u, BlasInt ldu, T* vt, BlasInt ldvt)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int gesdd<BlasReal>(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasReal* a, BlasInt lda, BlasReal* s, BlasReal* u, BlasInt ldu,
                           BlasReal* vt, BlasInt ldvt)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gesdd<BlasReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::sgesdd>(matrixOrder, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <>
inline int gesdd<BlasDoubleReal>(CBLAS_ORDER matrixOrder, char jobz, BlasInt m, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* s, BlasDoubleReal* u,
                                 BlasInt ldu, BlasDoubleReal* vt, BlasInt ldvt)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("gesdd<BlasDoubleReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::dgesdd>(matrixOrder, jobz, m, n, a, lda, s, u, ldu, vt, ldvt);
}

template <class T>
int laswp(CBLAS_ORDER matrixOrder, BlasInt n, T* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int laswp<BlasReal>(CBLAS_ORDER matrixOrder, BlasInt n, BlasReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("laswp<BlasReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::slaswp>(matrixOrder, n, a, lda, k1, k2, ipiv, incx);
}

template <>
inline int laswp<BlasDoubleReal>(CBLAS_ORDER matrixOrder, BlasInt n, BlasDoubleReal* a, BlasInt lda, BlasInt k1, BlasInt k2, const BlasInt* ipiv, BlasInt incx)
{
#ifdef INSTRUMENT_BLAS_CALLS
   INSTRUMENT_BLAS("laswp<BlasDoubleReal>");
#endif
   return BlasDispatcher::instance().call<details::BlasFunction::dlaswp>(matrixOrder, n, a, lda, k1, k2, ipiv, incx);
}

template <class T>
int getrs(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const T* a, BlasInt lda, const BlasInt* ipiv, T* b, BlasInt ldb)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int getrs<BlasReal>(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasReal* a, BlasInt lda, const BlasInt* ipiv, BlasReal* b,
                           BlasInt ldb)
{
   return BlasDispatcher::instance().call<details::BlasFunction::sgetrs>(matrix_order, trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <>
inline int getrs<BlasDoubleReal>(CBLAS_ORDER matrix_order, char trans, BlasInt n, BlasInt nrhs, const BlasDoubleReal* a, BlasInt lda, const BlasInt* ipiv,
                                 BlasDoubleReal* b, BlasInt ldb)
{
   return BlasDispatcher::instance().call<details::BlasFunction::dgetrs>(matrix_order, trans, n, nrhs, a, lda, ipiv, b, ldb);
}

template <class T>
int gels(CBLAS_ORDER matrix_layout, char trans, BlasInt m, BlasInt n, BlasInt nrhs, T* a, BlasInt lda, T* b, BlasInt ldb)
{
   ensure(0, "BlasInterface not implemented for this type!");
}

template <>
inline int gels<BlasReal>(CBLAS_ORDER matrix_layout, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasReal* a, BlasInt lda, BlasReal* b, BlasInt ldb)
{
   return BlasDispatcher::instance().call<details::BlasFunction::sgels>(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
}

template <>
inline int gels<BlasDoubleReal>(CBLAS_ORDER matrix_layout, char trans, BlasInt m, BlasInt n, BlasInt nrhs, BlasDoubleReal* a, BlasInt lda, BlasDoubleReal* b,
                                BlasInt ldb)
{
   return BlasDispatcher::instance().call<details::BlasFunction::dgels>(matrix_layout, trans, m, n, nrhs, a, lda, b, ldb);
}
}

DECLARE_NAMESPACE_END