#pragma once

/**
 @file

 This file defines array operations that summarizes values of an array along a particular dimension (axis)
 */

DECLARE_NAMESPACE_NLL

/**
@brief compute the return type of a function applied to an array
*/
template <class T, size_t N, class Config, class Function>
using function_axis_return_type = decltype((details::fake<Function>())(Array<T, N, Config>()));

/**
@brief compute the expected return type of a function applied to an array along a given axis

Not pretty but it works...
*/
template <class T, size_t N, class Config, class Function>
using axis_apply_fun_type =
typename Array<T, N, Config>::template rebind_dim<N - 1>::other::template rebind<function_axis_return_type<T, N, Config, Function>>::other;

/**
@brief Apply a "summarizing" function on the elements along <axis>

The type of the resulting array is determined by the return type of the summarizing function
*/
template <class T, size_t N, class Config, class Function>
axis_apply_fun_type<T, N, Config, Function> constarray_axis_apply_function(const Array<T, N, Config>& array, size_t axis, Function& f)
{
   static_assert(N >= 2, "can only do this for RANK >= 2");
   if (array.size() == 0)
   {
      return axis_apply_fun_type<T, N, Config, Function>();
   }

   StaticVector<ui32, N> min_index;
   StaticVector<ui32, N> max_index;

   // find the bounds, result shape and mapping result->array
   StaticVector<ui32, N - 1> shape_result;
   StaticVector<ui32, N - 1> mapping_result_array;
   size_t index = 0;
   for (ui32 n = 0; n < N; ++n)
   {
      if (n != axis)
      {
         shape_result[index]         = array.shape()[n];
         mapping_result_array[index] = n;
         ++index;
      }
      max_index[n] = array.shape()[n] - 1; // inclusive bounds
   }

   // for each value of <result>, iterate over the <axis> dimension
   // here we can only iterate over a single element at a time
   using array_result = axis_apply_fun_type<T, N, Config, Function>;
   array_result result(shape_result);

   using pointer_type       = typename array_result::pointer_type;
   using const_pointer_type = typename array_result::const_pointer_type;

   bool hasMoreElements = true;
   ArrayProcessor_contiguous_byMemoryLocality<array_result> iterator(result, 1);
   while (hasMoreElements)
   {
      pointer_type ptr(nullptr);
      hasMoreElements         = iterator.accessSingleElement(ptr);
      const auto currentIndex = iterator.getArrayIndex();

      for (size_t n = 0; n < N - 1; ++n)
      {
         max_index[mapping_result_array[n]] = currentIndex[n];
         min_index[mapping_result_array[n]] = currentIndex[n];
      }
      // finally apply the operation along <axis>
      const auto ref = const_cast<Array<T, N, Config>&>(array)(
          min_index,
          max_index); // for interface usability, the sub-array of a const array is not practical. Instead, unconstify the array and constify the reference
      const auto value = f(ref);
      details::copy_naive(ptr, 1, &value, 1, 1);
   }
   return result;
}

namespace details
{
/**
   @brief Simple adaptor for the mean function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_mean
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return mean(array);
   }
};

/**
   @brief Simple adaptor for the sum function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_sum
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return sum(array);
   }
};

/**
   @brief Simple adaptor for the max function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_max
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return max(array);
   }
};

/**
   @brief Simple adaptor for the min function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_min
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return min(array);
   }
};

/**
   @brief Simple adaptor for the argmin function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_argmin
{
   template <class T, size_t N, class Config>
   typename Array<T, N, Config>::index_type operator()(const Array<T, N, Config>& array) const
   {
      return argmin(array);
   }
};

/**
   @brief Simple adaptor for the min function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
struct adaptor_argmax
{
   template <class T, size_t N, class Config>
   typename Array<T, N, Config>::index_type operator()(const Array<T, N, Config>& array) const
   {
      return argmax(array);
   }
};

/**
   @brief Simple adaptor for the min function

   The purpose is to simplify the API: let the compiler figure out what is the type of the array
   */
template <class Predicate>
struct adaptor_count
{
   adaptor_count(const Predicate& predicate) : predicate(predicate)
   {
   }

   template <class T, size_t N, class Config>
   ui32 operator()(const Array<T, N, Config>& array) const
   {
      return count(array, predicate);
   }

   const Predicate& predicate;
};

/**
@brief Simple adaptor for the stddev function

The purpose is to simplify the API: let the compiler figure out what is the type of the array
*/
struct adaptor_stddev
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return stddev(array);
   }
};

/**
@brief Simple adaptor for the variance function

The purpose is to simplify the API: let the compiler figure out what is the type of the array
*/
struct adaptor_variance
{
   template <class T, size_t N, class Config>
   T operator()(const Array<T, N, Config>& array) const
   {
      return variance(array);
   }
};

}

/**
@brief return the mean value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_mean> mean(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_mean f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the sum of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_sum> sum(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_sum f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the variance of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_variance> variance(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_variance f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the standard deviation of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_stddev> stddev(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_stddev f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the max value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_max> max(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_max f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the min value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_min> min(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_min f;
   return constarray_axis_apply_function(array, axis, f);
}

/**
@brief return the min value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config, class Predicate>
axis_apply_fun_type<T, N, Config, details::adaptor_count<Predicate>> count(const Array<T, N, Config>& array, size_t axis, const Predicate& predicate)
{
   details::adaptor_count<Predicate> f(predicate);
   return constarray_axis_apply_function(array, axis, f);
}

namespace details
{
// custom version of <constarray_axis_apply_function> where we add the based index to the functor's result
// TODO: lot of code duplication, can we do better?
template <class T, size_t N, class Config, class Function>
axis_apply_fun_type<T, N, Config, Function> constarray_axis_apply_function_baseindex(const Array<T, N, Config>& array, size_t axis, Function& f)
{
   static_assert(N >= 2, "can only do this for RANK >= 2");
   if (array.size() == 0)
   {
      return axis_apply_fun_type<T, N, Config, Function>();
   }

   StaticVector<ui32, N> min_index;
   StaticVector<ui32, N> max_index;

   // find the bounds, result shape and mapping result->array
   StaticVector<ui32, N - 1> shape_result;
   StaticVector<ui32, N - 1> mapping_result_array;
   size_t index = 0;
   for (ui32 n = 0; n < N; ++n)
   {
      if (n != axis)
      {
         shape_result[index]         = array.shape()[n];
         mapping_result_array[index] = n;
         ++index;
      }
      max_index[n] = array.shape()[n] - 1; // inclusive bounds
   }

   // for each value of <result>, iterate over the <axis> dimension
   // here we can only iterate over a single element at a time
   using array_result = axis_apply_fun_type<T, N, Config, Function>;
   array_result result(shape_result);

   using pointer_type       = typename array_result::pointer_type;
   using const_pointer_type = typename array_result::const_pointer_type;

   bool hasMoreElements = true;
   ArrayProcessor_contiguous_byMemoryLocality<array_result> iterator(result, 1);
   while (hasMoreElements)
   {
      pointer_type ptr(nullptr);
      hasMoreElements         = iterator.accessSingleElement(ptr);
      const auto currentIndex = iterator.getArrayIndex();

      for (size_t n = 0; n < N - 1; ++n)
      {
         max_index[mapping_result_array[n]] = currentIndex[n];
         min_index[mapping_result_array[n]] = currentIndex[n];
      }
      // finally apply the operation along <axis>
      const auto ref = const_cast<Array<T, N, Config>&>(array)(
          min_index,
          max_index); // for interface usability, the sub-array of a const array is not practical. Instead, unconstify the array and constify the reference
      const auto value = f(ref) + min_index;
      details::copy_naive(ptr, 1, &value, 1, 1);
   }
   return result;
}
}

/**
@brief return the index of the the min value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_argmin> argmin(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_argmin f;
   return details::constarray_axis_apply_function_baseindex(array, axis, f);
}

/**
@brief return the index of the max value of all the elements contained in the array along a given axis
*/
template <class T, size_t N, class Config>
axis_apply_fun_type<T, N, Config, details::adaptor_argmax> argmax(const Array<T, N, Config>& array, size_t axis)
{
   details::adaptor_argmax f;
   return details::constarray_axis_apply_function_baseindex(array, axis, f);
}

DECLARE_NAMESPACE_NLL_END
