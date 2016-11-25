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
void add_naive(T* v1, const T* v2, size_t size)
{
   const T* end = v1 + size;
   for (; v1 != end; ++v1)
   {
      *v1 += *v2;
      ++v1;
      ++v2;
   }
}

template <class T>
void sub_naive(T* v1, const T* v2, size_t size)
{
   const T* end = v1 + size;
   for (; v1 != end; ++v1, ++v2)
   {
      *v1 -= *v2;
   }
}

template <class T>
void mul_naive(T* v1, const T value, size_t size)
{
   const T* end = v1 + size;
   for (; v1 != end; ++v1)
   {
      *v1 *= value;
   }
}

/**
    @brief compute v1 += v2 * mul
    */
template <class T>
void addmul_naive(T* v1, const T* v2, T mul, size_t size)
{
   const T* end = v1 + size;
   for (; v1 != end; ++v1)
   {
      *v1 += *v2 * mul;
      ++v1;
      ++v2;
   }
}

/**
    @brief compute sqrt(sum(v1^2))
    */
template <class T, class Accum = T>
Accum norm2_naive(const T* v1, size_t size)
{
   const auto accum = std::accumulate(v1, v1 + size, Accum());
   return std::sqrt(accum);
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