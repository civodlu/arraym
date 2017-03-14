#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief LU decomposition of a general matrix

Compute L, U such that A = L * U, where:
- A, m*n is a general matrix
- L (m*m) is lower triangular with unit diagonal elements (lower trapezoidal if m > n),
- U (m*n) is upper triangular (upper trapezoidal if m < n)
*/
template <class T, class Config>
void lu(const Matrix_BlasEnabled<T, 2, Config>& a, Array<T, 2, Config>& l_out, Array<T, 2, Config>& u_out)
{
   using matrix_type = Array<T, 2, Config>;
   using index_type  = typename matrix_type::index_type;

   const auto matrixOrder = getMatrixMemoryOrder(a);

   auto a_cpy = a; // make a copy as gesdd will destroy the content. Additionally it ensures there is no stride

   const blas::BlasInt lda = leading_dimension<T, Config>(a_cpy);
   const auto m            = static_cast<blas::BlasInt>(a.rows());
   const auto n            = static_cast<blas::BlasInt>(a.columns());
   const auto ipiv_size    = std::min(m, n);
   std::unique_ptr<blas::BlasInt> IPIV(new blas::BlasInt[ipiv_size]);

   const auto r = blas::getrf<T>(matrixOrder, m, n, &a_cpy(0, 0), lda, IPIV.get());
   ensure(r == 0, "error! blas::getrf<T>=" << r);

   l_out = matrix_type(index_type(m, m));
   u_out = matrix_type(index_type(m, n));
   for (int u = 0; u < m; ++u)
   {
      l_out(u, u) = 1;
      for (int v = 0; v < std::min(u, n); ++v)
      {
         l_out(u, v) = a_cpy(u, v);
      }
   }

   for (size_t u = 0; u < m; ++u)
   {
      for (size_t v = u; v < n; ++v)
      {
         u_out(u, v) = a_cpy(u, v);
      }
   }

   // Note:
   // - IPIV is 1-based indexe
   // - <?getrf> seems to compute P * A = L * U not A = P * L * U, or is it because <?laswp> is not doing what I expect it should be?
   // Should test this on other implementations...
   const blas::BlasInt llda = leading_dimension<T, Config>(l_out);
   const auto r2            = blas::laswp<T>(matrixOrder, m, &l_out(0, 0), llda, 1, ipiv_size, IPIV.get(), -1);
   ensure(r2 == 0, "error laswp");
}

DECLARE_NAMESPACE_NLL_END
