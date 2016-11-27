#pragma once

/**
 @file

 This file is implementing basic array & matrix calculations in the simplest manner. This may be useful as "default" implementation
 for types that are not supported by more specialized routines (e.g., BLAS, vectorization...)

 These other implementations must support exactly the same operations
 */
DECLARE_NAMESPACE_NLL

/**
 @brief Simplify the std::enable_if expression so that it is readable
 */
template <class T, int N, class Config>
using Array_NaiveEnabled = typename std::enable_if<array_use_naive<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

namespace details
{
/**
@brief Computes a1 += a2
*/
template <class T, int N, class Config, class Config2>
Array_NaiveEnabled<T, N, Config>& array_add(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   ensure(a1.shape() == a2.shape(), "must have the same shape!");
   ensure(same_data_ordering(a1, a2), "data must have a similar ordering!");

   // we MUST use processors: data may not be contiguous or with stride...
   ConstArrayProcessor_contiguous_byMemoryLocality<Array<T, N, Config2>> processor_a2(a2);
   ArrayProcessor_contiguous_byMemoryLocality<Array<T, N, Config>> processor_a1(a1);

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      T* ptr_a1       = nullptr;
      T const* ptr_a2 = nullptr;
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      hasMoreElements = processor_a2.accessMaxElements(ptr_a2);
      NLL_FAST_ASSERT(processor_a1.getMaxAccessElements() == processor_a2.getMaxAccessElements(), "memory line must have the same size");

      add_naive(ptr_a1, processor_a1.stride(), ptr_a2, processor_a2.stride(), processor_a1.getMaxAccessElements());
   }

   return a1;
}
}

DECLARE_NAMESPACE_END