#include "forward.h"
#include "wrapper-cublas.h"
#include "cublas-common.h"

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace blas
{
   template <>
   ARRAY_API BlasReal nrm2<BlasReal>(const BlasInt N, const cuda_ptr<BlasReal> X, const BlasInt incX)
   {
      BlasReal result = 0;
      INIT_AND_CHECK_CUDA(cublasSnrm2(detail::config.handle(), N, X, incX, &result));
      return result;
   }

   template <>
   ARRAY_API BlasDoubleReal nrm2<BlasDoubleReal>(const BlasInt N, const cuda_ptr<BlasDoubleReal> X, const BlasInt incX)
   {
      BlasDoubleReal result = 0;
      INIT_AND_CHECK_CUDA(cublasDnrm2(detail::config.handle(), N, X, incX, &result));
      return result;
   }

   template <>
   ARRAY_API BlasReal asum<BlasReal>(const BlasInt n, const cuda_ptr<BlasReal> x, const BlasInt incx)
   {
      BlasReal result = 0;
      INIT_AND_CHECK_CUDA(cublasSasum(detail::config.handle(), n, x, incx, &result));
      return result;
   }

   template <>
   ARRAY_API BlasDoubleReal asum<BlasDoubleReal>(const BlasInt n, const cuda_ptr<BlasDoubleReal> x, const BlasInt incx)
   {
      BlasDoubleReal result = 0;
      INIT_AND_CHECK_CUDA(cublasDasum(detail::config.handle(), n, x, incx, &result));
      return result;
   }

   template <>
   ARRAY_API BlasInt axpy<BlasReal>(BlasInt N, BlasReal alpha, const cuda_ptr<BlasReal> x, BlasInt incx, cuda_ptr<BlasReal> y, BlasInt incy)
   {
      INIT_AND_CHECK_CUDA(cublasSaxpy(detail::config.handle(), N, &alpha, x, incx, y, incy));
      return 0;
   }

   template <>
   ARRAY_API BlasInt axpy<BlasDoubleReal>(BlasInt N, BlasDoubleReal alpha, const cuda_ptr<BlasDoubleReal> x, BlasInt incx, cuda_ptr<BlasDoubleReal> y, BlasInt incy)
   {
      INIT_AND_CHECK_CUDA(cublasDaxpy(detail::config.handle(), N, &alpha, x, incx, y, incy));
      return 0;
   }

   template <>
   ARRAY_API BlasInt gemm<BlasReal>(
      CBLAS_ORDER matrixOrder,
      const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB,
      const BlasInt M, const BlasInt N, const BlasInt K,
      const BlasReal alpha, const cuda_ptr<BlasReal> A, const BlasInt lda,
      const cuda_ptr<BlasReal> B, const BlasInt ldb, const BlasReal beta,
      cuda_ptr<BlasReal> C, const BlasInt ldc)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      // run calculation on the GPU
      auto transa = detail::is_transposed(matrixOrder, TransA);
      auto transb = detail::is_transposed(matrixOrder, TransB);

      INIT_AND_CHECK_CUDA(cublasSgemm(detail::config.handle(), transa, transb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc));
      return 0;
   }

   template <>
   ARRAY_API BlasInt gemm<BlasDoubleReal>(
      CBLAS_ORDER matrixOrder,
      const enum CBLAS_TRANSPOSE TransA,
      const enum CBLAS_TRANSPOSE TransB,
      const BlasInt M, const BlasInt N, const BlasInt K,
      const BlasDoubleReal alpha, const cuda_ptr<BlasDoubleReal> A, const BlasInt lda,
      const cuda_ptr<BlasDoubleReal> B, const BlasInt ldb, const BlasDoubleReal beta,
      cuda_ptr<BlasDoubleReal> C, const BlasInt ldc)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      // run calculation on the GPU
      auto transa = detail::is_transposed(matrixOrder, TransA);
      auto transb = detail::is_transposed(matrixOrder, TransB);

      INIT_AND_CHECK_CUDA(cublasDgemm(detail::config.handle(), transa, transb, M, N, K, &alpha, A, lda, B, ldb, &beta, C, ldc));
      return 0;
   }

   template <>
   ARRAY_API BlasInt scal<BlasReal>(const BlasInt N, const BlasReal alpha, cuda_ptr<BlasReal> X, const BlasInt incX)
   {
      INIT_AND_CHECK_CUDA(cublasSscal(detail::config.handle(), N, &alpha, X, incX));
      return 0;
   }

   template <>
   ARRAY_API BlasInt scal<BlasDoubleReal>(const BlasInt N, const BlasDoubleReal alpha, cuda_ptr<BlasDoubleReal> X, const BlasInt incX)
   {
      INIT_AND_CHECK_CUDA(cublasDscal(detail::config.handle(), N, &alpha, X, incX));
      return 0;
   }

   template <>
   ARRAY_API BlasReal dot<BlasReal>(const BlasInt N, const cuda_ptr<BlasReal> x, const BlasInt incX, const cuda_ptr<BlasReal> y, const BlasInt incY)
   {
      BlasReal result = 0;
      INIT_AND_CHECK_CUDA(cublasSdot(detail::config.handle(), N, x, incX, y, incY, &result));
      return result;
   }

   template <>
   ARRAY_API BlasDoubleReal dot<BlasDoubleReal>(const BlasInt N, const cuda_ptr<BlasDoubleReal> x, const BlasInt incX, const cuda_ptr<BlasDoubleReal> y, const BlasInt incY)
   {
      BlasDoubleReal result = 0;
      INIT_AND_CHECK_CUDA(cublasDdot(detail::config.handle(), N, x, incX, y, incY, &result));
      return result;
   }

   template <>
   ARRAY_API BlasInt ger<BlasReal>(
      CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N,
      const BlasReal alpha, const cuda_ptr<BlasReal> x, const BlasInt incX,
      const cuda_ptr<BlasReal> y, const BlasInt incY,
      cuda_ptr<BlasReal> A, const BlasInt lda)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      INIT_AND_CHECK_CUDA(cublasSger(detail::config.handle(), M, N, &alpha, x, incX, y, incY, A, lda));
      return 0;
   }

   template <>
   ARRAY_API BlasInt ger<BlasDoubleReal>(
      CBLAS_ORDER matrixOrder, const BlasInt M, const BlasInt N,
      const BlasDoubleReal alpha, const cuda_ptr<BlasDoubleReal> x, const BlasInt incX,
      const cuda_ptr<BlasDoubleReal> y, const BlasInt incY,
      cuda_ptr<BlasDoubleReal> A, const BlasInt lda)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      INIT_AND_CHECK_CUDA(cublasDger(detail::config.handle(), M, N, &alpha, x, incX, y, incY, A, lda));
      return 0;
   }

   template <>
   ARRAY_API BlasInt gemv<BlasReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA,
      const BlasInt M, const BlasInt N, const BlasReal alpha,
      const cuda_ptr<BlasReal> A, const BlasInt lda,
      const cuda_ptr<BlasReal> x, const BlasInt incX, const BlasReal beta,
      cuda_ptr<BlasReal> y, const BlasInt incY)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      // run calculation on the GPU
      auto transa = detail::is_transposed(matrixOrder, TransA);

      INIT_AND_CHECK_CUDA(cublasSgemv(detail::config.handle(), transa, M, N, &alpha, A, lda, x, incX, &beta, y, incY));
      return 0;
   }

   template <>
   ARRAY_API BlasInt gemv<BlasDoubleReal>(CBLAS_ORDER matrixOrder, const enum CBLAS_TRANSPOSE TransA,
      const BlasInt M, const BlasInt N, const BlasDoubleReal alpha,
      const cuda_ptr<BlasDoubleReal> A, const BlasInt lda,
      const cuda_ptr<BlasDoubleReal> x, const BlasInt incX, const BlasDoubleReal beta,
      cuda_ptr<BlasDoubleReal> y, const BlasInt incY)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      // run calculation on the GPU
      auto transa = detail::is_transposed(matrixOrder, TransA);

      INIT_AND_CHECK_CUDA(cublasDgemv(detail::config.handle(), transa, M, N, &alpha, A, lda, x, incX, &beta, y, incY));
      return 0;
   }

   template <>
   ARRAY_API BlasInt getrf<float>(CBLAS_ORDER matrixOrder, BlasInt m, BlasInt n, cuda_ptr<float> a, BlasInt lda, cuda_ptr<BlasInt> ipiv)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");
      ensure(m == n, "CUBLAS handles only M == N");

      const int batch_size = 1;
      auto A_ptr2 = cuda::allocate_gpu<float*>(1);
      cudaMemcpy(A_ptr2.get(), &a.ptr, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice);

      auto info_array = cuda::allocate_gpu<int>(1);
      INIT_AND_CHECK_CUDA(cublasSgetrfBatched(detail::config.handle(), m, A_ptr2.get(), lda, ipiv, info_array.get(), batch_size));

      int result;
      cuda::vector_gpu_to_cpu(1, info_array.get(), 1, &result, 1);
      ensure(result == 0, "cublasSgetrfBatched failed!");
      return 0;
   }

   template <>
   ARRAY_API BlasInt getri<float>(CBLAS_ORDER matrixOrder, BlasInt n, cuda_ptr<float> a, BlasInt lda, const cuda_ptr<BlasInt> ipiv)
   {
      ensure(matrixOrder == CBLAS_ORDER::CblasColMajor, "CUBLAS handle only column major");

      const int batch_size = 1;

      // CUBLAS can't do the inversion inplace
      auto A_ptr2 = cuda::allocate_gpu<float*>(1);
      cudaMemcpy(A_ptr2.get(), &a, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice);

      auto c = cuda::allocate_gpu<float>(n * n);
      auto C_ptr2 = cuda::allocate_gpu<float*>(1);
      cudaMemcpy(C_ptr2.get(), &c, sizeof(float*), cudaMemcpyKind::cudaMemcpyHostToDevice);

      const int ldc = n;
      auto info_array = cuda::allocate_gpu<int>(1);

      float** A_ptr3 = A_ptr2.get();
      float** C_ptr3 = C_ptr2.get();
      INIT_AND_CHECK_CUDA(cublasSgetriBatched(detail::config.handle(), n, (const float**)A_ptr3, lda, ipiv, C_ptr3, ldc, info_array.get(), batch_size));

      int result;
      cuda::vector_gpu_to_cpu(1, info_array.get(), 1, &result, 1);
      ensure(result == 0, "cublasSgetrfBatched failed!");
      return 0;
   }
}

DECLARE_NAMESPACE_NLL_END

#endif // WITH_CUDA
