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

template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
T norm2sqr(const T& v)
{
   return v * v;
}

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
void add_naive_cte(T* v1, size_t stride_v1, size_t size, T value)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1)
   {
      *v1 += value;
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
   NLL_FAST_ASSERT(value != T(0), "div by 0");
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1)
   {
      *v1 /= value;
   }
}

/**
 @brief Compute v1[e] /= v2[e] elementwise
 */
template <class T, class T2>
void div_naive_elementwise(T* v1, size_t stride_v1, const T2* v2, size_t stride_v2, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 /= *v2;
   }
}

/**
@brief Compute v1[e] *= v2[e] elementwise
*/
template <class T, class T2>
void mul_naive_elementwise(T* v1, size_t stride_v1, const T2* v2, size_t stride_v2, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 *= *v2;
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
@brief compute v1 = static_cast<T>(v2)
*/
template <class T, class T2>
void static_cast_naive(T* v1, size_t stride_v1, const T2* v2, size_t stride_v2, size_t size)
{
   const T* end = v1 + size * stride_v1;
   for (; v1 != end; v1 += stride_v1, v2 += stride_v2)
   {
      *v1 = static_cast<T>(*v2);
   }
}

/**
@brief compute sum(v1^2)
*/
template <class T, class Accum = T>
Accum norm2_naive_sqr(const T* v1, size_t stride_v1, size_t nb_elements)
{
   const T* end = v1 + nb_elements * stride_v1;
   Accum accum  = 0;
   for (; v1 != end; v1 += stride_v1)
   {
      accum += norm2sqr(*v1);
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
@brief Write a strided array to a stream
*/
template <class T>
void write_naive(const T* y_pointer, ui32 y_stride, ui32 nb_elements, std::ostream& f)
{
   if (y_stride == 1)
   {
      // if no stride, we can write the whole buffer at once
      const size_t size_bytes = nb_elements * sizeof(T);
      f.write(reinterpret_cast<const char*>(y_pointer), size_bytes);
   }
   else
   {
      // else we need to individually write each element
      const T* y_end = y_pointer + y_stride * nb_elements;
      for (; y_pointer != y_end; y_pointer += y_stride)
      {
         f.write(reinterpret_cast<const char*>(y_pointer), sizeof(T));
      }
   }
}

/**
@brief Read a strided array to a stream
*/
template <class T>
void read_naive(T* y_pointer, ui32 y_stride, ui32 nb_elements, std::istream& f)
{
   if (y_stride == 1)
   {
      // if no stride, we can write the whole buffer at once
      const size_t size_bytes = nb_elements * sizeof(T);
      f.read(reinterpret_cast<char*>(y_pointer), size_bytes);
   }
   else
   {
      // else we need to individually write each element
      T* y_end = y_pointer + y_stride * nb_elements;
      for (; y_pointer != y_end; y_pointer += y_stride)
      {
         f.read(reinterpret_cast<char*>(y_pointer), sizeof(T));
      }
   }
}

/**
@brief y = f(x)

Apply a function on a strided array x and assign it to another strided array y
*/
template <class T, class T2, class F>
void apply_naive2(T* y_pointer, ui32 y_stride, const T2* x_pointer, ui32 x_stride, ui32 nb_elements, F& f)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride, x_pointer += x_stride)
   {
      *y_pointer = f(*x_pointer);
   }
}

/**
@brief f(y, x)

Apply a function on a strided array x and assign it to another strided array y
*/
template <class T1, class T2, class F>
void apply_naive2_const(const T1* y_pointer, ui32 y_stride, const T2* x_pointer, ui32 x_stride, ui32 nb_elements, F& f)
{
   const T1* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride, x_pointer += x_stride)
   {
      f(*y_pointer, *x_pointer);
   }
}

/**
@brief y = f(y)

*/
template <class T, class F>
void apply_naive1(T* y_pointer, ui32 y_stride, ui32 nb_elements, F& f)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride)
   {
      *y_pointer = f(*y_pointer);
   }
}

/**
@brief apply f(y), no result
*/
template <class T, class F>
void apply_naive1_const(T const* y_pointer, ui32 y_stride, ui32 nb_elements, F& f)
{
   const T* y_end = y_pointer + y_stride * nb_elements;
   for (; y_pointer != y_end; y_pointer += y_stride)
   {
      f(*y_pointer);
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

/**
@brief In a strided array, return the max index
*/
template <class T>
std::pair<ui32, T> argmax(const T* ptr_start, ui32 stride, ui32 nb_elements)
{
   ui32 index       = 0;
   T max_value      = ptr_start[0];
   const T* end_ptr = ptr_start + nb_elements * stride;
   for (const T* ptr = ptr_start; ptr != end_ptr; ptr += stride)
   {
      auto value = *ptr;
      if (max_value < value)
      {
         max_value = value;
         index     = static_cast<ui32>((ptr - ptr_start) / stride);
      }
   }
   return std::make_pair(index, max_value);
}

/**
@brief In a strided array, return the min index
*/
template <class T>
std::pair<ui32, T> argmin(const T* ptr_start, ui32 stride, ui32 nb_elements)
{
   ui32 index       = 0;
   T min_value      = ptr_start[0];
   const T* end_ptr = ptr_start + nb_elements * stride;
   for (const T* ptr = ptr_start; ptr != end_ptr; ptr += stride)
   {
      auto value = *ptr;
      if (min_value > value)
      {
         min_value = value;
         index     = static_cast<ui32>((ptr - ptr_start) / stride);
      }
   }
   return std::make_pair(index, min_value);
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
 @brief return 1 if value is >= 0, -1 otherwise
 */
template <class T>
T sign(T value)
{
   static_assert(std::is_signed<T>::value, "must be a signed type!");
   return value >= 0 ? T(1) : T(-1);
}

/**
@ingroup core
@brief test if 2 values are equal with a certain tolerance
*/
template <class T, typename = typename std::enable_if<std::is_floating_point<T>::value>::type>
bool equal(const T val1, const T val2, const T tolerance = 2 * std::numeric_limits<T>::epsilon())
{
   return std::abs(val1 - val2) <= tolerance;
}

template <class T, typename = int, typename = typename std::enable_if<std::is_integral<T>::value>::type>
bool equal(const T val1, const T val2, const T UNUSED(tolerance) = 0)
{
   return val1 == val2;
}

template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
T sqr(const T& v)
{
   return v * v;
}

/**
@return if value > max return max
        if value < min return min
        else return value
*/
template <class Toutput, class Tinput, typename = typename std::enable_if<std::is_arithmetic<Tinput>::value>::type>
Toutput saturate_value(const Tinput value, const Tinput min_value, const Tinput max_value)
{
   return static_cast<Toutput>((value < min_value) ? min_value : ((value > max_value) ? max_value : value));
}

DECLARE_NAMESPACE_NLL_END
