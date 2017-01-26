#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace blas
{
   template <class T>
   BlasInt axpy(BlasInt N, T alpha, const cuda_ptr<T> x, BlasInt incx, cuda_ptr<T> y, BlasInt incy);
}

DECLARE_NAMESPACE_NLL_END

#endif