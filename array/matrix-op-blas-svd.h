#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief svd(X) produces a diagonal matrix S of the same dimension as a, with nonnegative diagonal elements in decreasing order,
and unitary matrices U and V so that a = u * s * vt
@param u The first min(m, n) columns of u are the left singular vectors of a
@param v The first min(m, n) rows of vt are the right singular vectors of a
@param s the singular values stored in decreasing order
*/
template <class T, class Config, class Config2>
void svd(const Matrix_BlasEnabled<T, 2, Config>& a, Array<T, 2, Config>& u, Vector<T, Config2>& s, Array<T, 2, Config>& vt)
{
   static_assert(std::is_same<Matrix_BlasEnabled<T, 2, Config>, Array<T, 2, Config>>::value, "matrix type is not supported for BLAS");
   using matrix_type = Array<T, 2, Config>;
   using vector_type = Vector<T, Config2>;
   matrix_type a_cpy = a; // make a copy, gesdd will destroy the content

   const auto matrixOrder = getMatrixMemoryOrder(a);

   const int m      = (int)a.rows();
   const int n      = (int)a.columns();
   const int s_size = std::min(m, n);

   u  = matrix_type({m, m});
   vt = matrix_type({n, n});
   s  = vector_type(s_size);

   const blas::BlasInt lda = leading_dimension<T, Config>(a_cpy);
   blas::BlasInt ldu       = leading_dimension<T, Config>(u);
   blas::BlasInt ldvt      = leading_dimension<T, Config>(vt);
   const auto success      = blas::gesdd<T>(matrixOrder, 'A', m, n, &a_cpy(0, 0), lda, &s(0), &u(0, 0), ldu, &vt(0, 0), ldvt);
   ensure(success == 0, "gesdd failed!");

   // cusolverDnSgesvd
}

/**
@brief utility function to reconstruct <s> from a vector
*/
template <class T, class Config, class Config2>
Matrix_BlasEnabled<T, 2, Config> svd_construct_s(const Array<T, 2, Config>& u, const Vector<T, Config2>& s, const Array<T, 2, Config>& vt)
{
   Array<T, 2, Config> s_matrix(u.columns(), vt.rows());
   for (size_t n = 0; n < s.size(); ++n)
   {
      s_matrix(n, n) = s(n);
   }
   return s_matrix;
}

DECLARE_NAMESPACE_NLL_END
