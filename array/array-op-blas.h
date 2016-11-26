#pragma once

/**
@file

This file is implementing basic array & matrix calculations in the simplest manner. This may be useful as "default" implementation
for types that are not supported by more specialized routines (e.g., BLAS, vectorization...)

These other implementations must support exactly the same operations
*/
DECLARE_NAMESPACE_NLL

/**
@brief Alias to simplify the use of std::enable_if for array_use_blas arrys
*/
template <class T, int N, class Config>
using Array_BlasEnabled = typename std::enable_if<array_use_blas<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

namespace details
{
   template <class T, int N, class Config>
   Array_BlasEnabled<T, N, Config>& array_add( Array<T, N, Config>& a1, const Array<T, N, Config>& a2 )
   {
      ensure( same_data_ordering( a1, a2 ), "data must have a similar ordering!" );

      Array<T, N, Config> cpy = a1;
      return a1;
   }
}

DECLARE_NAMESPACE_END