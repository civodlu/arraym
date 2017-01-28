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

/**
 @brief return a copy of array with std::cos applied to each element
 */
template <class T, size_t N, class Config>
Array<T, N, Config> cos(const Array<T, N, Config>& array)
{
   auto op = [](T value)
   {
      return std::cos(value);
   };
   return constarray_apply_function(array, op);
}

/**
@brief return a copy of array with std::sin applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sin(const Array<T, N, Config>& array)
{
   auto op = [](T value)
   {
      return std::sin(value);
   };
   return constarray_apply_function(array, op);
}

/**
@brief return a copy of array with std::sqrt applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sqrt(const Array<T, N, Config>& array)
{
   auto op = [](T value)
   {
      return std::sqrt(value);
   };
   return constarray_apply_function(array, op);
}

/**
@brief return a copy of array with for each element e is returned e * e
*/
template <class T, size_t N, class Config>
Array<T, N, Config> sqr( const Array<T, N, Config>& array )
{
   auto op = []( T value )
   {
      return value * value;
   };
   return constarray_apply_function( array, op );
}

/**
@brief return a copy of array with std::abs applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> abs(const Array<T, N, Config>& array)
{
   auto op = [](T value)
   {
      return std::abs(value);
   };
   return constarray_apply_function(array, op);
}

/**
@brief return a copy of array with std::log applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> log( const Array<T, N, Config>& array )
{
   auto op = []( T value )
   {
      return std::log( value );
   };
   return constarray_apply_function( array, op );
}

/**
@brief return a copy of array with std::exp applied to each element
*/
template <class T, size_t N, class Config>
Array<T, N, Config> exp( const Array<T, N, Config>& array )
{
   auto op = []( T value )
   {
      return std::exp( value );
   };
   return constarray_apply_function( array, op );
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
   T max_value = std::numeric_limits<T>::min();
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

DECLARE_NAMESPACE_NLL_END