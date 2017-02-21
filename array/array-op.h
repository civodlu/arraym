#pragma once

DECLARE_NAMESPACE_NLL

/**
 @brief return a copy of the array where each element is transformed by a given function @p f

 @tparam Function must be callable with (T value)
 */
template <class T, size_t N, class Config, class Function>
Array<T, N, Config> constarray_apply_function(const Array<T, N, Config>& array, Function& f)
{
   using array_type = Array<T, N, Config>;
   using pointer_type = typename array_type::pointer_type;
   using const_pointer_type = typename array_type::const_pointer_type;

   static_assert(is_callable_with<Function, T>::value, "Op is not callable!");
   auto op = [&](pointer_type a1_pointer, ui32 a1_stride, const_pointer_type a2_pointer, ui32 a2_stride, ui32 nb_elements)
   {
      details::apply_naive2(a1_pointer, a1_stride, a2_pointer, a2_stride, nb_elements, f);
   };

   Array<T, N, Config> array_cpy(array.shape());
   iterate_array_constarray(array_cpy, array, op);
   return array_cpy;
}

/**
@brief return a copy of the array where each element is transformed by a given function @p f

@tparam Function must be callable (pointer_type a1_pointer, ui32 a1_stride, const_pointer_type a2_pointer, ui32 a2_stride, ui32 nb_elements)
*/
template <class T, size_t N, class Config, class Function>
Array<T, N, Config> constarray_apply_function_strided_array(const Array<T, N, Config>& array, Function& f)
{
   using array_type = Array<T, N, Config>;
   using pointer_type = typename array_type::pointer_type;
   using const_pointer_type = typename array_type::const_pointer_type;

   static_assert(is_callable_with<Function, pointer_type, ui32, const_pointer_type, ui32, ui32>::value, "Op is not callable!");
   Array<T, N, Config> array_cpy(array.shape());
   iterate_array_constarray(array_cpy, array, f);
   return array_cpy;
}

/**
@brief Simply apply op(array)
@tparam Function must be callable with (T value)
*/
template <class T, size_t N, class Config, class Op>
void constarray_apply_function_inplace(const Array<T, N, Config>& array, Op& op)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;

   auto f = [&](T value)
   {
      op(value);
   };

   auto op_constarray = [&](const_pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
   {
      details::apply_naive1_const(a1_pointer, a1_stride, nb_elements, f);
   };
   iterate_constarray(array, op_constarray);
}

namespace details
{
   //
   // Here we want to expose the function as a strided array. This is to enable more custom implementations
   // using overloads (e.g., CUDA gpu)
   //
   
   template <class T>
   void cos(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::cos(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void sin(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::sin(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void sqrt(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::sqrt(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void sqr(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return value * value;
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void abs(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::abs(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void exp(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::exp(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   template <class T>
   void log(T* output, ui32 output_stride, const T* input, ui32 input_stride, ui32 nb_elements)
   {
      auto op = [](T value)
      {
         return std::log(value);
      };
      apply_fun_array_strided(output, output_stride, input, input_stride, nb_elements, op);
   }

   //
   // TODO min, max, mean. Problems: how to combine multiple data segments?
   //
}

/**
 @brief return a copy of array with std::cos applied to each element
 */
template <class T, size_t N, class Config>
Array<T, N, Config> cos(const Array<T, N, Config>& array)
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::cos<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with std::sin applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sin(const Array<T, N, Config>& array)
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::sin<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with std::sqrt applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sqrt(const Array<T, N, Config>& array)
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::sqrt<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with for each element e is returned e * e
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sqr( const Array<T, N, Config>& array )
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::sqr<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with std::abs applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> abs(const Array<T, N, Config>& array)
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::abs<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with std::log applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> log( const Array<T, N, Config>& array )
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::log<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return a copy of array with std::exp applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> exp( const Array<T, N, Config>& array )
{
   void(*ptr)(T*, ui32, const T*, ui32, ui32) = &details::exp<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

/**
@brief return the min value contained in the array
*/
template <class T, size_t N, class Config>
T min(const Array<T, N, Config>& array)
{
   T min_value = std::numeric_limits<T>::max();
   auto f = [&](T value)
   {
      min_value = std::min(min_value, value);
   };
   constarray_apply_function_inplace(array, f);
   return min_value;
}

/**
@brief return the max value contained in the array
*/
template <class T, size_t N, class Config>
T max(const Array<T, N, Config>& array)
{
   T max_value = std::numeric_limits<T>::lowest();
   auto f = [&](T value)
   {
      max_value = std::max(max_value, value);
   };
   constarray_apply_function_inplace(array, f);
   return max_value;
}

/**
@brief return the mean value of all the elements contained in the array
*/
template <class T, size_t N, class Config, class Accum = T>
Accum sum( const Array<T, N, Config>& array )
{
   Accum accum = 0;
   auto f = [&]( T value )
   {
      accum += value;
   };
   constarray_apply_function_inplace( array, f );
   return accum;
}

/**
@brief return the mean value of all the elements contained in the array
*/
template <class T, size_t N, class Config, class Accum = T>
Accum mean( const Array<T, N, Config>& array )
{
   return sum(array) / static_cast<T>(array.size());
}

template <class T, size_t N, class Config>
typename PromoteFloating<T>::type norm2(const Array<T, N, Config>& a1)
{
   using const_pointer_type = typename Array<T, N, Config>::const_pointer_type;
   using return_type = typename PromoteFloating<T>::type;
   return_type accum = 0;
   auto op = [&](const_pointer_type ptr, ui32 stride, ui32 elements)
   {
      accum += details::norm2_naive_sqr(ptr, stride, elements);
   };

   iterate_constarray(a1, op);
   return std::sqrt(accum);
}

/**
 @brief Stack arrays of a same shape into a higher dimensional array

 Input arrays are stored at the last dimension
 */
template <class T, size_t N, class Config, typename... Other>
Array<T, N + 1, typename Config::template rebind_dim<N + 1>::other> stack(const Array<T, N, Config>& array, const Other&... other)
{
   using array_type = Array<T, N, Config>;
   using result_type = Array<T, N + 1, typename Config::template rebind_dim<N + 1>::other>;
   using array_ref = ArrayRef<T, N, Config>;
   static_assert(is_same_nocvr<array_type, Other...>::value, "arguments must all be of the same type!");

   // pack everything in an array for easy manipulation
   const Array<T, N, Config>* arrays[] =
   {
      &array,
      &other...
   };

   const size_t nb_array = sizeof...(Other)+1;
   for (auto index : range<size_t>(1, nb_array))
   {
      ensure(arrays[0]->shape() == arrays[index]->shape(), "all arrays must be of the same shape");
   }

   // precompute some variables
   typename result_type::index_type min_index_result;
   typename result_type::index_type result_size;
   for (auto index : range(N))
   {
      result_size[index] = arrays[0]->shape()[index];
   }
   result_size[N] = nb_array;

   // copy slice by slice the arrays
   result_type s(result_size);
   for (auto index : range<ui32>(0, nb_array))
   {
      min_index_result[N] = index;
      auto slice = s.template slice<N>(min_index_result);
      slice = (*arrays[index]);
   }
   return s;
}

/**
@brief Return the index of the max element of the array
*/
template <class T, size_t N, class Config>
typename Array<T, N, Config>::index_type argmax(const Array<T, N, Config>& array)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;
   ConstArrayProcessor_contiguous_byMemoryLocality<array_type> processor_a1(array, 0);

   T max_value = std::numeric_limits<T>::lowest();
   typename array_type::index_type max_value_index;

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      const_pointer_type ptr_a1(nullptr);
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      auto pair_index_max = details::argmax(ptr_a1, processor_a1.stride(), processor_a1.getNbElementsPerAccess());
      if (pair_index_max.second > max_value)
      {
         max_value = pair_index_max.second;
         max_value_index = processor_a1.getArrayIndex();
         max_value_index[processor_a1.getVaryingIndex()] += pair_index_max.first;
      }
   }

   return max_value_index;
}

/**
@brief Return the index of the min element of the array
*/
template <class T, size_t N, class Config>
typename Array<T, N, Config>::index_type argmin(const Array<T, N, Config>& array)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;
   ConstArrayProcessor_contiguous_byMemoryLocality<array_type> processor_a1(array, 0);

   T min_value = std::numeric_limits<T>::max();
   typename array_type::index_type min_value_index;

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      const_pointer_type ptr_a1(nullptr);
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      auto pair_index_min = details::argmin(ptr_a1, processor_a1.stride(), processor_a1.getNbElementsPerAccess());
      if (pair_index_min.second < min_value)
      {
         min_value = pair_index_min.second;
         min_value_index = processor_a1.getArrayIndex();
         min_value_index[processor_a1.getVaryingIndex()] += pair_index_min.first;
      }
   }

   return min_value_index;
}

/**
@brief count the number of times the predicate if true
*/
template <class T, size_t N, class Config, class Predicate>
ui32 count( const Array<T, N, Config>& array, const Predicate& predicate )
{
   ui32 c = 0;
   auto f = [&]( T value )
   {
      if ( predicate( value ) )
      {
         ++c;
      }
   };

   constarray_apply_function_inplace( array, f );
   return c;
}

DECLARE_NAMESPACE_NLL_END