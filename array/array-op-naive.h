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
    @brief iterate array & const array jointly
    @tparam must be callable using (T* a1_pointer, a1_stride, const T* a2_pointer, a2_stride, nb elements)
    */
template <class T, class T2, int N, class Config, class Config2, class Op>
void iterate_array_constarray(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2, const Op& op)
{
   static_assert(is_callable_with<Op, T*, ui32, const T2*, ui32, ui32>::value, "Op is not callable!");

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

      op(ptr_a1, processor_a1.stride(), ptr_a2, processor_a2.stride(), processor_a1.getMaxAccessElements());
   }
}

/**
   @brief iterate array
   @tparam must be callable using (T* a1_pointer, a1_stride, nb elements)
   */
template <class T, int N, class Config, class Op>
void iterate_array(Array<T, N, Config>& a1, const Op& op)
{
   ArrayProcessor_contiguous_byMemoryLocality<Array<T, N, Config>> processor_a1(a1);

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      T* ptr_a1       = nullptr;
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      op(ptr_a1, processor_a1.stride(), processor_a1.getMaxAccessElements());
   }
}

/**
@brief Computes a1 += a2
*/
template <class T, int N, class Config, class Config2>
Array_NaiveEnabled<T, N, Config>& array_add(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   auto op = &add_naive<T>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 -= a2
*/
template <class T, int N, class Config, class Config2>
Array_NaiveEnabled<T, N, Config>& array_sub(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   auto op = &sub_naive<T>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 *= cte
*/
template <class T, int N, class Config>
Array_NaiveEnabled<T, N, Config>& array_mul(Array<T, N, Config>& a1, T a2)
{
   auto op = [&](T* ptr, ui32 stride, ui32 elements) { mul_naive(ptr, stride, a2, elements); };

   iterate_array(a1, op);
   return a1;
}

/**
 @brief Computes a1 /= cte
 @note we can't just use array_mul with 1/a2 for integral types so use a div function
 */
template <class T, int N, class Config>
Array_NaiveEnabled<T, N, Config>& array_div(Array<T, N, Config>& a1, T a2)
{
   auto op = [&](T* ptr, ui32 stride, ui32 elements) { div_naive(ptr, stride, a2, elements); };

   iterate_array(a1, op);
   return a1;
}

/**
 @brief Display matrices
 */
template <class T, size_t N, class Config, typename = typename std::enable_if<is_matrix<Array<T, N, Config>>::value>::type>
std::ostream& operator<<(std::ostream& o, const Array<T, N, Config>& array)
{
   if (array.size() == 0)
   {
      o << "[]";
      return o;
   }

   o << "[ ";
   for (size_t r = 0; r < array.rows(); ++r)
   {
      for (size_t c = 0; c < array.columns(); ++c)
      {
         o << array(r, c) << " ";
      }
      if (r + 1 < array.rows())
      {
         o << std::endl << "  ";
      }
   }
   o << "]";
   return o;
}

/**
@brief Display any other array type than matrices
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type, typename = int>
std::ostream& operator<<(std::ostream& o, const Array<T, N, Config>& array)
{
   if (array.size() == 0)
   {
      o << "[]";
      return o;
   }

   o << "[ ";

   ConstArrayProcessor_contiguous_byDimension<Array<T, N, Config>> processor(array);
   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      T const* ptr     = nullptr;
      hasMoreElements  = processor.accessSingleElement(ptr);
      const auto index = processor.getArrayIndex()[0];
      o << *ptr;
      if (index == 0 && hasMoreElements)
      {
         o << "\n ";
      }
      else
      {
         o << ' ';
      }
   }

   o << "]";
   return o;
}
}

DECLARE_NAMESPACE_END