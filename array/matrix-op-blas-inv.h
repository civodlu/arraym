#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief return the inverse of the input
@note Do no throw if the matrix is singular. Instead, the returned size if (0,0)
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> inv_nothrow(const Array<T, 2, Config>& a)
{
   using matrix_type = Array<T, 2, Config>;
   ensure(a.rows() == a.columns(), "must be square!");
   matrix_type i = a;

   const auto size = static_cast<blas::BlasInt>(a.rows());
   std::unique_ptr<blas::BlasInt> IPIV(new blas::BlasInt[size + 1]);

   const auto memory_order = getMatrixMemoryOrder(a);
   blas::BlasInt lda;
   const auto& stride_a = i.getMemory().getIndexMapper()._getPhysicalStrides();
   if (memory_order == CBLAS_ORDER::CblasColMajor)
   {
      lda = stride_a[1];
      ensure(stride_a[0] == 1, "can't have stride != 1 ");
   }
   else
   {
      lda = stride_a[0];
      ensure(stride_a[1] == 1, "can't have stride != 1 ");
   }

   const auto r = core::blas::getrf<T>(memory_order, size, size, &i(0, 0), lda, IPIV.get());
   if (r != 0)
   {
      // something is wrong... just return an empty array
      return matrix_type();
   }

   const auto r2 = core::blas::getri<T>(memory_order, size, &i(0, 0), lda, IPIV.get());
   if (r2 != 0)
   {
      // something is wrong... just return an empty array
      return matrix_type();
   }

   return i;
}

/**
@brief return the inverse of the input

Throw an exception if the inverse failed
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> inv(const Array<T, 2, Config>& a)
{
   auto r = inv_nothrow(a);
   ensure(r.size() != 0, "inv failed!");
   return r;
}

DECLARE_NAMESPACE_NLL_END