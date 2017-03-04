#pragma once

DECLARE_NAMESPACE_NLL
//
// Vector  OP VECTOR
//

template <class T, size_t N>
inline StaticVector<T, N>& operator+=(StaticVector<T, N>& lhs, const StaticVector<T, N>& rhs)
{
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] += rhs[n];
   }
   return lhs;
}

template <class T, size_t N>
inline StaticVector<T, N>& operator-=(StaticVector<T, N>& lhs, const StaticVector<T, N>& rhs)
{
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] -= rhs[n];
   }
   return lhs;
}

template <class T, size_t N>
inline StaticVector<T, N> operator+(const StaticVector<T, N>& lhs, const StaticVector<T, N>& rhs)
{
   StaticVector<T, N> res(lhs);
   return res += rhs;
}

template <class T, size_t N>
inline StaticVector<T, N> operator-(const StaticVector<T, N>& lhs, const StaticVector<T, N>& rhs)
{
   StaticVector<T, N> res(lhs);
   return res -= rhs;
}

//
// OP Vector
//

template <class T, size_t N>
inline StaticVector<T, N> operator-(const StaticVector<T, N>& lhs)
{
   StaticVector<T, N> res;
   for (size_t n = 0; n < N; ++n)
      res[n]     = -lhs[n];
   return res;
}

//
// Vector  OP cte
//

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator+=(StaticVector<T, N>& lhs, T2 rhs)
{
   const T val = static_cast<T>(rhs);
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] += val;
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator+(const StaticVector<T, N>& lhs, T2 rhs)
{
   StaticVector<T, N> res(lhs);
   return res += rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator-=(StaticVector<T, N>& lhs, T2 rhs)
{
   const T val = static_cast<T>(rhs);
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] -= val;
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator-(const StaticVector<T, N>& lhs, T2 rhs)
{
   StaticVector<T, N> res(lhs);
   return res -= rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N>& operator*=(StaticVector<T, N>& lhs, const T2& rhs)
{
   // Do NOT cast to T => we might be multipling by a float...
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] = static_cast<T>(lhs[n] * rhs);
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator*(const StaticVector<T, N>& lhs, const T2& rhs)
{
   StaticVector<T, N> res(lhs);
   return res *= rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N>& operator/=(StaticVector<T, N>& lhs, const T2& rhs)
{
   // Do NOT cast to T => we might be dividing by a float...
   NLL_FAST_ASSERT(rhs != 0, "div by 0");
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] = static_cast<T>(lhs[n] / rhs);
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator/(const StaticVector<T, N>& lhs, const T2& rhs)
{
   StaticVector<T, N> res(lhs);
   return res /= rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N>& operator/=(StaticVector<T, N>& lhs, const StaticVector<T2, N>& rhs)
{
   // Do NOT cast to T => we might be dividing by a float...
   for (size_t n = 0; n < N; ++n)
   {
      NLL_FAST_ASSERT(rhs[n] != 0, "div by 0");
      lhs[n] /= static_cast<T>(rhs[n]);
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator/(const StaticVector<T, N>& lhs, const StaticVector<T2, N>& rhs)
{
   StaticVector<T, N> res(lhs);
   return res /= rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N>& operator*=(const T2& rhs, StaticVector<T, N>& lhs)
{
   // Do NOT cast to T => we might be multipling by a float...
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] = static_cast<T>(lhs[n] * rhs);
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator*(const T2& rhs, const StaticVector<T, N>& lhs)
{
   StaticVector<T, N> res(lhs);
   return res *= rhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N>& operator*=(StaticVector<T, N>& lhs, const StaticVector<T2, N>& rhs)
{
   // Do NOT cast to T => we might be dividing by a float...
   for (size_t n = 0; n < N; ++n)
   {
      lhs[n] *= static_cast<T>(rhs[n]);
   }
   return lhs;
}

template <class T, size_t N, typename T2, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
inline StaticVector<T, N> operator*(const StaticVector<T, N>& lhs, const StaticVector<T2, N>& rhs)
{
   StaticVector<T, N> res(lhs);
   return res *= rhs;
}

DECLARE_NAMESPACE_NLL_END
