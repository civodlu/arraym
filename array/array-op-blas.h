#pragma once

/**
@file

This file is implementing basic array & matrix calculations in the simplest manner. This may be useful as "default" implementation
for types that are not supported by more specialized routines (e.g., BLAS, vectorization...)

These other implementations must support exactly the same operations
*/
DECLARE_NAMESPACE_NLL

/**
@brief Alias to simplify the use of std::enable_if for array_use_blas arrays
*/
template <class T, size_t N, class Config>
using Array_BlasEnabled = typename std::enable_if<array_use_blas<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

namespace details
{
/**
 @brief computes Y += a * X using BLAS
 */
template <class T, size_t N, class Config, class Config2>
void axpy(T a, const Array<T, N, Config>& x, Array<T, N, Config2>& y)
{
   ensure(x.shape() == y.shape(), "must have the same shape!");
   ensure(same_data_ordering(x, y), "data must have a similar ordering!");

   using index_type = typename Array<T, N, Config>::index_type;
   if (is_array_fully_contiguous(x) && is_array_fully_contiguous(y))
   {
      // the two array are using contiguous memory with no gap at all, so we can just
      // use BLAS on the array's memory all at once
      T const* ptr_x = &x(index_type());
      T* ptr_y       = &y(index_type());
      blas::axpy<T>(static_cast<blas::BlasInt>(x.size()), a, ptr_x, 1, ptr_y, 1);
   }
   else
   {
      auto op = [&](T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements) {
         blas::axpy<T>(static_cast<blas::BlasInt>(nb_elements), a, x_pointer, x_stride, y_pointer, y_stride);
      };
      iterate_array_constarray(y, x, op);
   }
}

/**
@brief computes Y *= a using BLAS
*/
template <class T, size_t N, class Config>
void scal(Array<T, N, Config>& y, T a)
{
   using index_type = typename Array<T, N, Config>::index_type;
   if (is_array_fully_contiguous(y))
   {
      // the two array are using contiguous memory with no gap at all, so we can just
      // use BLAS on the array's memory all at once
      T* ptr_y = &y(index_type());
      blas::scal<T>(static_cast<blas::BlasInt>(y.size()), a, ptr_y, 1);
   }
   else
   {
      auto op = [&](T* y_pointer, ui32 y_stride, ui32 nb_elements) { blas::scal<T>(static_cast<blas::BlasInt>(nb_elements), a, y_pointer, y_stride); };
      iterate_array(y, op);
   }
}

template <class T, size_t N, class Config, class Config2>
Array_BlasEnabled<T, N, Config>& array_add(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   axpy(static_cast<T>(1), a2, a1);
   return a1;
}

template <class T, size_t N, class Config, class Config2>
Array_BlasEnabled<T, N, Config>& array_sub(Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   axpy(static_cast<T>(-1), a2, a1);
   return a1;
}

template <class T, size_t N, class Config>
Array_BlasEnabled<T, N, Config>& array_mul(Array<T, N, Config>& a1, T value)
{
   scal(a1, value);
   return a1;
}

template <class T, size_t N, class Config>
Array_BlasEnabled<T, N, Config>& array_div(Array<T, N, Config>& a1, T a2)
{
   return array_mul(a1, static_cast<T>(1) / a2);
}
}

DECLARE_NAMESPACE_END