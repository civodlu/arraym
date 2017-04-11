#pragma once

DECLARE_NAMESPACE_NLL

template <class Type, size_t N>
StaticVector<Type, N> sqrt(const StaticVector<Type, N>& v)
{
   StaticVector<Type, N> r(no_init_tag); // make sure the vector is not initialized
   for (size_t n = 0; n < N; ++n)
   {
      r[n] = std::sqrt(v[n]);
   }
   return r;
}

template <class Type, size_t N>
StaticVector<Type, N> sqr(const StaticVector<Type, N>& v)
{
   StaticVector<Type, N> r(no_init_tag); // make sure the vector is not initialized
   for (size_t n = 0; n < N; ++n)
   {
      r[n] = v[n] * v[n];
   }
   return r;
}

template <class Type, size_t N, class TypeSum = Type>
TypeSum sum(const StaticVector<Type, N>& v)
{
   return std::accumulate(v.begin(), v.end(), TypeSum());
}

/**
 @brief round each element to the nearest integer
*/
template <class result_type, class T, size_t N>
StaticVector<result_type, N> round(const StaticVector<T, N>& v)
{
   StaticVector<result_type, N> r(no_init_tag);
   for (size_t n = 0; n < N; ++n)
   {
      r[n] = static_cast<result_type>(std::round(v[n]));
   }
   return r;
}

/**
@ingroup core
@brief Computes the outer product of two 3-vectors
*/
template <class T>
StaticVector<T, 3> cross(const StaticVector<T, 3>& a, const StaticVector<T, 3>& b)
{
   StaticVector<T, 3> res(no_init_tag);
   res[0] = a[1] * b[2] - a[2] * b[1];
   res[1] = a[2] * b[0] - a[0] * b[2];
   res[2] = a[0] * b[1] - a[1] * b[0];
   return res;
}

template <class T, size_t N>
T dot(const StaticVector<T, N>& lhs, const StaticVector<T, N>& rhs)
{
   T accum = 0;
   for (size_t n = 0; n < N; ++n)
      accum += lhs[n] * rhs[n];
   return accum;
}

template <class T, size_t N>
T norm2sqr(const StaticVector<T, N>& lhs)
{
   T accum = 0;
   for (size_t n = 0; n < N; ++n)
      accum += lhs[n] * lhs[n];
   return accum;
}

template <class T, size_t N>
typename PromoteFloating<T>::type norm2(const StaticVector<T, N>& lhs)
{
   return static_cast<typename PromoteFloating<T>::type>(std::sqrt(norm2sqr(lhs)));
}

template <class T, typename = typename std::enable_if<std::is_arithmetic<T>::value>::type>
T norm2(const T& lhs)
{
   return std::abs(lhs);
}

DECLARE_NAMESPACE_NLL_END
