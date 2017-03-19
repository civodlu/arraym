#pragma once

DECLARE_NAMESPACE_NLL

/**
   @brief return a diagonal matrix where <args> represent the elements of its diagonal
   */
template <class T, typename... Args, typename = typename std::enable_if<!is_same_nocvr<Vector<T>, typename first<Args...>::type>::value>::type>
Matrix<T> diag( Args&&... args )
{
   const T vals[] = { static_cast<T>( args )... };

   const auto dim = sizeof...(Args);
   Matrix<T> m({ dim, dim });
   for (ui32 n = 0; n < dim; ++n)
   {
      m(n, n) = vals[n];
   }
   return m;
}

/**
@brief return a diagonal matrix where <args> represent the elements of its diagonal
*/
template <class T>
Matrix<T> diag( const Vector<T>& vals )
{
   const auto size = vals.size();
   Matrix<T> m({ size, size });
   for (ui32 n = 0; n < size; ++n)
   {
      m(n, n) = vals[n];
   }
   return m;
}

DECLARE_NAMESPACE_NLL_END
