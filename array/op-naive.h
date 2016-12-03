#pragma once

/**
 @file

 In this file is implemented basic math operations. These are basic building blocks the library is relying on.
 In particular here, it is implemented in a generic way (i.e., should work for any integral/floating type) but it will
 not be efficient. More refined implementation can be used for more specific type. 

 We should as much as possible use these building blocks so that once more specialized operations are implemented,
 (e.g., loop enrolling, SSE, BLAS..) it will benefit the whole library.
*/

DECLARE_NAMESPACE_NLL

namespace details
{
template <class T>
void add_naive(T* v1, size_t stride_v1, const T* v2, const size_t stride_v2, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 += *v2;
   }
}

template <class T>
void sub_naive(T* v1, size_t stride_v1, const T* v2, size_t stride_v2, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 -= *v2;
   }
}

template <class T>
void mul_naive(T* v1, size_t stride_v1, const T value, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1)
   {
      *v1 *= value;
   }
}

template <class T>
void div_naive(T* v1, size_t stride_v1, const T value, size_t size)
{
   NLL_FAST_ASSERT(value != 0, "div by 0");
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1)
   {
      *v1 /= value;
   }
}

/**
    @brief compute v1 += v2 * mul
    */
template <class T>
void addmul_naive(T* v1, size_t stride_v1, const T* v2, size_t stride_v2, T mul, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 -= *v2 * mul;
   }
}

/**
@brief compute sum(v1^2)
*/
template <class T, class Accum = T>
Accum norm2_naive_sqr( T const* v1, size_t stride_v1, size_t nb_elements )
{
   const T* end = v1 + nb_elements * stride_v1;
   Accum accum = 0;
   for ( ; v1 != end; v1 += stride_v1)
   {
      accum += *v1 * *v1;
   }
   return accum;
}

/**
 @brief y = x

 Copy from a strided array x to another strided array y
 */
template <class T>
void copy_naive(T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements)
{
   if (y_stride == 1 && x_stride == 1)
   {
      const size_t size_bytes = nb_elements * sizeof(T);
      memcpy(y_pointer, x_pointer, size_bytes);
   }
   else 
   {
      const T* y_end = y_pointer + y_stride * nb_elements;
      for (; y_pointer != y_end; y_pointer += y_stride, x_pointer += x_stride)
      {
         *y_pointer = *x_pointer;
      }
   }
}

/**
@brief y = f(x)

Apply a function on a strided array x and assign it to another strided array y
*/
template <class T, class F>
void apply_naive2(T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements, const F& f)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride, x_pointer += x_stride)
   {
      *y_pointer = f(*x_pointer);
   }
}

/**
@brief y = f(y)

*/
template <class T, class F>
void apply_naive1(T* y_pointer, ui32 y_stride, ui32 nb_elements, const F& f)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride)
   {
      *y_pointer = f(*y_pointer);
   }
}

/**
@brief y = value

*/
template <class T>
void set_naive(T* y_pointer, ui32 y_stride, ui32 nb_elements, T value)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride)
   {
      *y_pointer = value;
   }
}

}

template <class T, class Accum = T>
Accum dot(const T* v1, const T* v2, size_t size)
{
   Accum accum = Accum();
   for (size_t n = 0; n < size; ++n)
   {
      accum += v1[n] * v2[n];
   }
   return accum;
}

/**
@ingroup core
@brief calculate an absolute value. Work around for Visual and ambiguous case...
*/
inline double absolute(double val)
{
   return val >= 0 ? val : -val;
}

/**
@ingroup core
@brief test if 2 values are equal with a certain tolerance
*/
template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
bool equal(const T val1, const T val2, const T tolerance = 2 * std::numeric_limits<T>::epsilon())
{
   return absolute((double)val1 - (double)val2) <= (double)tolerance;
}

DECLARE_NAMESPACE_END