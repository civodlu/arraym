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
         T* ptr_a1 = nullptr;
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
         T* ptr_a1 = nullptr;
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
   auto op = [&](T* ptr, ui32 stride, ui32 elements)
   {
      mul_naive(ptr, stride, a2, elements);
   };

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
   auto op = [&](T* ptr, ui32 stride, ui32 elements)
   {
      div_naive(ptr, stride, a2, elements);
   };

   iterate_array(a1, op);
   return a1;
}

/**
 @brief Matrix * Matrix operation
 
 Very slow version... should probably always avoid it!
 */
template <class T, class Config, class Config2>
Array_NaiveEnabled<T, 2, Config> array_mul_array(const Array<T, 2, Config>& op1, const Array<T, 2, Config2>& op2)
{
   const size_t op2_sizex = op2.shape()[1];
   const size_t op1_sizex = op1.shape()[1];
   const size_t op1_sizey = op1.shape()[0];
   
   Array<T, 2, Config> m({ op1_sizey, op2_sizex });
   for (size_t nx = 0; nx < op2_sizex; ++nx)
   {
      for (size_t ny = 0; ny < op1_sizey; ++ny)
      {
         T val = 0;
         for (size_t n = 0; n < op1_sizex; ++n)
         {
            val += op1(ny, n) * op2(n, nx);
         }
         m(ny, nx) = val;
      }
   }
   return m;
}
}

DECLARE_NAMESPACE_END