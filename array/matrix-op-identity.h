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

DECLARE_NAMESPACE_END