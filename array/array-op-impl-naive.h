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
template <class T, size_t N, class Config>
using Array_NaiveEnabled = typename std::enable_if<array_use_naive<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

namespace details
{
/**
@brief Computes a1 += a2
*/
template <class T, size_t N, class Config, class Config2>
Array_NaiveEnabled<T, N, Config>& array_add(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   auto op = &add_naive<T>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 += a2
*/
template <class T, size_t N, class Config>
Array<T, N, Config>& array_add_cte(Array<T, N, Config>& a1, T a2)
{
   using pointer_type = typename Array<T, N, Config>::pointer_type;
   auto op            = [&](pointer_type ptr, ui32 stride, ui32 elements) { add_naive_cte<T>(ptr, stride, elements, a2); };
   iterate_array(a1, op);
   return a1;
}

/**
@brief Computes a1 -= a2
*/
template <class T, size_t N, class Config, class Config2>
Array_NaiveEnabled<T, N, Config>& array_sub(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   auto op = &sub_naive<T>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 *= cte
*/
template <class T, size_t N, class Config>
Array_NaiveEnabled<T, N, Config>& array_mul(Array<T, N, Config>& a1, T a2)
{
   using pointer_type = typename Array<T, N, Config>::pointer_type;
   auto op            = [&](pointer_type ptr, ui32 stride, ui32 elements) { mul_naive(ptr, stride, a2, elements); };

   iterate_array(a1, op);
   return a1;
}

/**
 @brief Computes a1 /= cte
 @note we can't just use array_mul with 1/a2 for integral types so use a div function
 */
template <class T, size_t N, class Config>
Array_NaiveEnabled<T, N, Config>& array_div(Array<T, N, Config>& a1, T a2)
{
   using pointer_type = typename Array<T, N, Config>::pointer_type;
   auto op            = [&](pointer_type ptr, ui32 stride, ui32 elements) { div_naive(ptr, stride, a2, elements); };

   iterate_array(a1, op);
   return a1;
}

/**
@brief Computes a1 /= a2, element by element
*/
template <class T, class T2, size_t N, class Config, class Config2>
Array<T, N, Config>& array_div_elementwise(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2)
{
   ensure(a1.shape() == a2.shape(), "must have the same shape!");
   auto op = &div_naive_elementwise<T, T2>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 /= a2, element by element
*/
template <class T, class T2, size_t N, class Config, class Config2>
Array<T, N, Config>& array_mul_elementwise_inplace(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2)
{
   ensure(a1.shape() == a2.shape(), "must have the same shape!");
   auto op = &mul_naive_elementwise<T, T2>;
   iterate_array_constarray(a1, a2, op);
   return a1;
}

/**
@brief Computes a1 /= a2, element by element
*/
template <class T, class T2, size_t N, class Config, class Config2>
Array<T, N, Config> array_mul_elementwise(const Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2)
{
   auto copy = a1;
   array_mul_elementwise_inplace(copy, a2);
   return copy;
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

   o << "[";

   const size_t max_nb_elements = 50;

   ConstArrayProcessor_contiguous_byDimension<Array<T, N, Config>> processor(array, 1);
   bool hasMoreElements = true;
   size_t nb_element = 0;
   while (hasMoreElements && nb_element < max_nb_elements)
   {
      T const* ptr    = nullptr;
      hasMoreElements = processor.accessSingleElement(ptr);
      o << *ptr;
      auto index = processor.getArrayIndex();
      if (index[0] + 1 == array.shape()[0] && hasMoreElements)
      {
         o << '\n';
      }
      if (hasMoreElements)
      {
         o << ' ';
      }
      ++nb_element;
   }

   if (nb_element >= max_nb_elements)
   {
      o << " ...";
   }

   o << "]";
   return o;
}
}

DECLARE_NAMESPACE_NLL_END
