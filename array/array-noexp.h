#pragma once

/**
 @file

 This file defines the array operators for the non enabled expression template arrays
 */
DECLARE_NAMESPACE_NLL

/**
@brief Simplify the std::enable_if expression so that it is readable
*/
template <class T, size_t N, class Config>
using Array_NaiveOperatorEnabled = typename std::enable_if<array_use_naive_operator<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1>& operator+=(Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs)
{
   details::array_add(lhs, rhs);
   return lhs;
}

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1> operator+(const Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs)
{
   Array<T, N, Config1> cpy = lhs;
   cpy += rhs;
   return cpy;
}

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1>& operator-=(Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs)
{
   details::array_sub(lhs, rhs);
   return lhs;
}

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1> operator-(const Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs)
{
   Array<T, N, Config1> cpy = lhs;
   cpy -= rhs;
   return cpy;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1>& operator*=(Array<T, N, Config1>& lhs, T2 value)
{
   details::array_mul(lhs, static_cast<T>( value ));
   return lhs;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1> operator*(Array<T, N, Config1>& lhs, T2 value)
{
   Array<T, N, Config1> cpy = lhs;
   cpy *= static_cast<T>(value);
   return cpy;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1>& operator+=(Array<T, N, Config1>& lhs, T2 value)
{
   details::array_add_cte(lhs, static_cast<T>(value));
   return lhs;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1> operator+(const Array<T, N, Config1>& lhs, T2 value)
{
   Array<T, N, Config1> cpy = lhs;
   cpy += static_cast<T>(value);
   return cpy;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1>& operator-=(Array<T, N, Config1>& lhs, T2 value)
{
   details::array_add_cte(lhs, static_cast<T>(-value));
   return lhs;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1> operator-(const Array<T, N, Config1>& lhs, T2 value)
{
   Array<T, N, Config1> cpy = lhs;
   cpy -= static_cast<T>(value);
   return cpy;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1> operator*(T2 value, Array<T, N, Config1>& rhs)
{
   Array<T, N, Config1> cpy = rhs;
   cpy *= static_cast<T>(value);
   return cpy;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1>& operator/=(Array<T, N, Config1>& lhs, T2 value)
{
   details::array_div(lhs, static_cast<T>(value));
   return lhs;
}

template <class T, class T2, size_t N, class Config1, typename = typename std::enable_if<std::is_convertible<T2, T>::value>::type>
Array_NaiveOperatorEnabled<T, N, Config1> operator/(Array<T, N, Config1>& lhs, T2 value)
{
   Array<T, N, Config1> cpy = lhs;
   cpy /= static_cast<T>(value);
   return cpy;
}

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1> operator*(const Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs)
{
   return array_mul_array(lhs, rhs);
}

template <class T, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, 2, Config1> operator*(const Array<T, 2, Config1>& lhs, const Array<T, 1, Config2>& rhs)
{
   using matrix_type = Array<T, 2, Config1>;
   using vector_type = Array<T, 1, Config2>;
   static_assert(IsArrayLayoutContiguous<vector_type>::value, "TODO handle rhs not a Memory_contiguous");
   
   typename matrix_type::Memory memory({ rhs.size(), 1 }, const_cast<T*>(&rhs(0)));
   matrix_type rhs_2d(memory);
   return array_mul_array(lhs, rhs_2d);
}

DECLARE_NAMESPACE_NLL_END
