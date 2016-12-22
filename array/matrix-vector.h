#pragma once

/**
 @file

 Utility functions to convert array to vector and vice versa
 */
DECLARE_NAMESPACE_NLL

/**
 @brief View the 1D vector as a matrix row major. The data is shared between the two
 */
template <class T, class Allocator>
Matrix_row_major<T, Allocator> as_matrix_row_major(const Vector<T, Allocator>& v, const typename Matrix_row_major<T, Allocator>::index_type& shape)
{
   using matrix_type = Matrix_row_major<T, Allocator>;
   matrix_type matrix(typename matrix_type::Memory(shape, const_cast<T*>(&v(0)), v.getMemory().getAllocator()));
   return matrix;
}

/**
@brief View the 1D vector as a matrix column major. The data is shared between the two
*/
template <class T, class Allocator>
Matrix_column_major<T, Allocator> as_matrix_column_major(const Vector<T, Allocator>& v, const typename Matrix_column_major<T, Allocator>::index_type& shape)
{
   using matrix_type = Matrix_column_major<T, Allocator>;
   matrix_type matrix(typename matrix_type::Memory(shape, const_cast<T*>(&v(0)), v.getMemory().getAllocator()));
   return matrix;
}

/**
@brief View the N-array as a 1D vector. The data is shared between the two
*/
template <class T, size_t N, class Config>
Vector<T, typename Config::allocator_type> as_vector(const Array<T, N, Config>& a)
{
   using vector_type = Vector<T, typename Config::allocator_type>;
   using array_index_type = typename Array<T, N, Config>::index_type;
   using vector_index_type = typename vector_type::index_type;
   vector_type v(typename vector_type::Memory(
      vector_index_type(a.size()),
      const_cast<T*>(&a(array_index_type())),
      a.getMemory().getAllocator()));
   return v;
}

DECLARE_NAMESPACE_NLL_END