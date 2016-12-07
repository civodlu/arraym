#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief identity matrix
*/
template <class T>
Matrix<T> identity( size_t rank )
{
   Matrix<T> m( static_cast<ui32>( rank ), static_cast<ui32>( rank ) );
   for ( size_t n = 0; n < rank; ++n )
   {
      m( n, n ) = 1;
   }
   
   return m;
}

/**
@brief fill the matrix as identity
*/
template <class T, class Mapper, class Allocator>
void identity( Matrix<T, Mapper, Allocator>& m )
{
   using matrix_type = Matrix<T, Mapper, Allocator>;
   ensure( m.rows() == m.columns(), "must be square!" );
   
   auto op = []( const matrix_type::index_type& i )
   {
      return static_cast<T>(i[ 0 ] == i[ 1 ]);
   };
   fill( m, op );
}

DECLARE_NAMESPACE_END