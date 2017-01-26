#include "forward.h"
#include "cublas_v2.h"

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace blas
{
   template <>
   ARRAY_API BlasInt axpy<BlasReal>(BlasInt N, BlasReal alpha, const cuda_ptr<BlasReal> x, BlasInt incx, cuda_ptr<BlasReal> y, BlasInt incy)
   {
#ifdef INSTRUMENT_BLAS_CALLS
      INSTRUMENT_BLAS("cublas.axpy<BlasReal>");
#endif
      /*
      cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n,
      const float           *alpha,
      const float           *x, int incx,
      float                 *y, int incy)
      */
      ensure(0, "TODO");
   }
}

DECLARE_NAMESPACE_NLL_END

#endif // WITH_CUDA
