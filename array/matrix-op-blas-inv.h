#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief compute the least square solution min_x ||B - A * X||_2   using QR decomposition

if a_transposed == true, solves min_x ||B - A^t * X||_2 instead
@return B
*/
template <class T, class Config>
Matrix_BlasEnabled<T, 2, Config> least_square(const Array<T, 2, Config>& a, const Array<T, 2, Config>& b, bool a_transposed = false)
{
   using matrix_type = Array<T, 2, Config>;
   const auto memory_order_a = getMatrixMemoryOrder(a);
   const auto memory_order_b = getMatrixMemoryOrder(b);
   ensure(memory_order_a == memory_order_b, "matrix must have the same memory order");


   auto a_cpy = a;  // will have the result of the factorization
   auto b_cpy = b;

   blas::BlasInt lda;
   blas::BlasInt ldb;
   const auto& stride_a = a.getMemory().getIndexMapper()._getPhysicalStrides();
   const auto& stride_b = a.getMemory().getIndexMapper()._getPhysicalStrides();
   if (memory_order_a == MatrixMemoryOrder::COLUMN_MAJOR)
   {
      lda = stride_a[1];
      ldb = stride_b[1];
   }
   else
   {
      lda = stride_a[0];
      ldb = stride_b[0];
   }

   const auto m = static_cast<blas::BlasInt>(a.rows());
   const auto n = static_cast<blas::BlasInt>(a.columns());
   const auto nrhs = static_cast<blas::BlasInt>(b.columns());

   //const auto info = blas::gels<T>(blas::CBLAS_ORDER::CblasColMajor, a_transposed ? 'T' : 'N', m, n, nrhs, &a_cpy(0, 0), lda, &a_cpy(0, 0), ldb);
   
   // TODO REFACTORING using blas::CBLAS_ORDER
   const auto info = blas::gels<T>(memory_order_a, a_transposed ? 'T' : 'N', m, n, nrhs, &a_cpy(0, 0), lda, &a_cpy(0, 0), ldb);
   ensure(info == 0, "blas::gels<T> failed!");

   return b_cpy({ 0, 0 }, {n - 1, nrhs - 1});
   //matrix_type X(n, nrhs);
   //X(core::rangeAll, core::rangeAll) = b_cpy(Range(0, n - 1), Range(0, nrhs - 1));
   //return X;
}

DECLARE_NAMESPACE_END