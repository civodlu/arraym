#pragma once

DECLARE_NAMESPACE_NLL

/**
 @brief Return the indices of the array where the predicate is true
 */
template <class T, size_t N, class Config, class Predicate>
std::vector<StaticVector<ui32, N>> where(const Array<T, N, Config>& array, Predicate p)
{
   std::vector<StaticVector<ui32, N>> indexes;
   any<T, N, Config, Predicate>(array, p, nullptr, &indexes);
   return indexes;
}

DECLARE_NAMESPACE_NLL_END