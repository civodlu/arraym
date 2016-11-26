#pragma once

/**
 @file

 This file is implementing basic array & matrix calculations in the simplest manner. This may be useful as "default" implementation
 for types that are not supported by more specialized routines (e.g., BLAS, vectorization...)
 */
DECLARE_NAMESPACE_NLL

/**
    @brief returns true if two array have similar data ordering. (i.e., using an iterator, we point to the same
           index for both arrays)
    */
template <class T, int N, class Config>
bool same_data_ordering(Array<T, N, Config>& a1, const Array<T, N, Config>& a2)
{
   // TODO this will not work with strided data! use only physical stride ordering & size
   return a1.getMemory().getIndexMapper()._getPhysicalStrides() == a2.getMemory().getIndexMapper()._getPhysicalStrides();
}

template <class T, int N, class Config, class = typename std::enable_if<array_use_naive<Array<T, N, Config>::value>>::type>
Array<T, N, Config>& operator+=(Array<T, N, Config>& a1, Array<T, N, Config>& a2)
{
   ensure( same_data_ordering( a1, a2 ), "data must have a similar ordering!" );
}

DECLARE_NAMESPACE_END