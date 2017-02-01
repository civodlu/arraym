#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace details
{
   template <class T>
   void cos(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sin(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sqrt(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void sqr(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void abs(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void exp(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   void log(cuda_ptr<T> output, ui32 output_stride, const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   T max(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T>
   T min(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);

   template <class T, class Accum=T>
   Accum sum(const cuda_ptr<T> input, ui32 input_stride, ui32 nb_elements);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> cos(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::cos<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> sin(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::sin<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> exp(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::exp<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> log(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::log<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> sqr(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::sqr<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> sqrt(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::sqrt<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> abs(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   void(*ptr)(cuda_ptr<T>, ui32, const cuda_ptr<T>, ui32, ui32) = &details::abs<T>;
   return constarray_apply_function_strided_array(array, ptr);
}

template <class T, size_t N, class Allocator>
T max(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   using array_type = Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>;
   using const_pointer_type = typename array_type::const_pointer_type;

   T(*ptr)(const cuda_ptr<T>, ui32, ui32) = &details::max<T>;
   
   T value = std::numeric_limits<T>::lowest();
   auto op_constarray = [&](const_pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
   {
      // here run each portion of the array independently, then we need to compute the merged result
      value = std::max(value, ptr(a1_pointer, a1_stride, nb_elements));
   };
   iterate_constarray(array, op_constarray);
   return value;
}

template <class T, size_t N, class Allocator>
T min(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   using array_type = Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>;
   using const_pointer_type = typename array_type::const_pointer_type;

   T(*ptr)(const cuda_ptr<T>, ui32, ui32) = &details::min<T>;

   T value = std::numeric_limits<T>::max();
   auto op_constarray = [&](const_pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
   {
      // here run each portion of the array independently, then we need to compute the merged result
      value = std::min(value, ptr(a1_pointer, a1_stride, nb_elements));
   };
   iterate_constarray(array, op_constarray);
   return value;
}

template <class T, size_t N, class Allocator, class Accum = T>
Accum sum(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array)
{
   using array_type = Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>;
   using const_pointer_type = typename array_type::const_pointer_type;

   T(*ptr)(const cuda_ptr<T>, ui32, ui32) = &details::sum<T, Accum>;

   Accum value = 0;
   auto op_constarray = [&](const_pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
   {
      // here run each portion of the array independently, then we need to compute the merged result
      auto r = ptr(a1_pointer, a1_stride, nb_elements);
      value += r;
   };
   iterate_constarray(array, op_constarray);
   return value;
}

DECLARE_NAMESPACE_NLL_END

#endif // WITH_CUDA