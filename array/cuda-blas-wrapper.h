#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace blas
{
   //
   // This header MUST be in sync with "blas-wrapper.h" (BLAS only, not LAPACK) and extended with cuda_ptr overloads
   //
   template <class T>
   T nrm2(const BlasInt N, const cuda_ptr<T> X, const BlasInt incX);

   template <class T>
   T asum(const BlasInt n, const cuda_ptr<T> x, const BlasInt incx);

   template <class T>
   BlasInt axpy(BlasInt N, T alpha, const cuda_ptr<T> x, BlasInt incx, cuda_ptr<T> y, BlasInt incy);

   template <class T>
   BlasInt gemm(
      CBLAS_ORDER matrixOrder,
      const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB,
      const BlasInt M, const BlasInt N, const BlasInt K,
      const T alpha, const cuda_ptr<T> A, const BlasInt lda,
      const cuda_ptr<T> B, const BlasInt ldb, const T beta,
      cuda_ptr<T> C, const BlasInt ldc);

   template <class T>
   BlasInt scal(const BlasInt N, const T alpha, cuda_ptr<T> X, const BlasInt incX);

   template <class T>
   T dot(const BlasInt N, const cuda_ptr<T> x, const BlasInt incX, const cuda_ptr<T> y, const BlasInt incY);
   
   template <class T>
   BlasInt ger(
      CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N,
      const T alpha, const cuda_ptr<T> x, const BlasInt incX,
      const cuda_ptr<T> y, const BlasInt incY,
      cuda_ptr<T> A, const BlasInt lda);

   template <class T>
   BlasInt gemv(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA,
      const BlasInt M, const BlasInt N, const T alpha,
      const cuda_ptr<T> A, const BlasInt lda,
      const cuda_ptr<T> x, const BlasInt incX, const T beta,
      cuda_ptr<T> y, const BlasInt incY);

   template <class T>
   BlasInt getrf(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, cuda_ptr<T> a, BlasInt lda, cuda_ptr<BlasInt> ipiv);

   template <class T>
   BlasInt getri(CBLAS_ORDER matrixOrder, BlasInt n, cuda_ptr<T> a, BlasInt lda, const cuda_ptr<BlasInt> ipiv);
}

DECLARE_NAMESPACE_NLL_END

#endif