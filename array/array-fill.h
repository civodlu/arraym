#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Generic fill of an array. The index order is defined by memory locality
@param functor will be called using functor(index_type(x, y, z, ...)), i.e., each coordinate components

Note: there will be a significan't overhead (as compared to naive access) due to the linear index to array index conversion
*/
template <class T, size_t N, class Config, class Functor>
void fill_index(Array<T, N, Config>& array, Functor functor)
{
   using functor_return = typename function_traits<Functor>::return_type;
   static_assert(std::is_same<functor_return, T>::value, "functor return type must be the same as array type");

   if (array.isEmpty())
   {
      return;
   }

   using array_type = Array<T, N, Config>;
   bool hasMoreElements = true;
   using pointer_type = typename array_type::pointer_type;

   ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(array, 1);
   while (hasMoreElements)
   {
      pointer_type ptr(nullptr);
      hasMoreElements = iterator.accessSingleElement(ptr);
      const auto currentIndex = iterator.getArrayIndex();

      const auto value = functor(currentIndex);
      details::copy_naive(ptr, 1, &value, 1, 1);
   }
}

namespace details
{
   /**
    @brief For each element of a strided array, apply a functor and copy the result to output
    */
   template <class T1, class T2, class F>
   void apply_fun_array_strided(T1* output, ui32 output_stride, const T2* input, ui32 input_stride, ui32 nb_elements, F f)
   {
      static_assert(is_callable_with<F, T2>::value, "Op is not callable with the correct arguments!");

      const T2* input_end = input + input_stride * nb_elements;
      for (; input != input_end; input += input_stride, output += output_stride)
      {
         *output = f(*input);
      }
   };
}

/**
@brief Generic fill of an array. The index order is defined by memory locality
@param functor will be called using functor(array::value_type), i.e., the value of the currently pointed element
*/
template <class T, size_t N, class Config, class Functor>
void fill_value(Array<T, N, Config>& array, Functor functor)
{
   using array_type = Array<T, N, Config>;
   using functor_return = typename function_traits<Functor>::return_type;
   using pointer_type = typename array_type::pointer_type;
   using const_pointer_type = typename array_type::const_pointer_type;
   static_assert(std::is_same<functor_return, T>::value, "functor return type must be the same as array type");

   if (array.isEmpty())
   {
      return;
   }

   // each iteration, access max elements. This way we remove most of the overhead. Then
   // for each value, apply the functor
   auto f = [&](pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
   {
      details::apply_fun_array_strided(a1_pointer, a1_stride, a1_pointer, a1_stride, nb_elements, functor);
   };

   iterate_array(array, f);
}

DECLARE_NAMESPACE_NLL_END