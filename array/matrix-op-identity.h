#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief identity matrix
*/
template <class T>
Matrix<T> identity(size_t rank)
{
   Matrix<T> m(static_cast<ui32>(rank), static_cast<ui32>(rank));
   for (size_t n = 0; n < rank; ++n)
   {
      m(n, n) = 1;
   }

   return m;
}

/**
@brief identity matrix
*/
template <class MatrixType>
MatrixType identityMatrix(size_t rank)
{
   MatrixType m({rank, rank});
   for (size_t n = 0; n < rank; ++n)
   {
      m(n, n) = 1;
   }

   return m;
}

/**
@brief fill the matrix as identity
*/
template <class T, class Config>
void identity(Array<T, 2, Config>& m)
{
   ensure(m.shape()[0] == m.shape()[1], "must be square!");

   auto op = [](const typename Array<T, 2, Config>::index_type& i) { return static_cast<T>(i[0] == i[1]); };
   fill_index(m, op);
}

DECLARE_NAMESPACE_NLL_END
