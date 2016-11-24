#pragma once

DECLARE_NAMESPACE_NLL

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