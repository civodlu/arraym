#pragma once

/**
 @file

 This file defines the array operators for the non enabled expression template arrays
 */
DECLARE_NAMESPACE_NLL

/**
@brief Simplify the std::enable_if expression so that it is readable
*/
template <class T, int N, class Config>
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
Array_NaiveOperatorEnabled<T, N, Config1>& operator-=( Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs )
{
   details::array_sub( lhs, rhs );
   return lhs;
}

template <class T, size_t N, class Config1, class Config2>
Array_NaiveOperatorEnabled<T, N, Config1> operator-( const Array<T, N, Config1>& lhs, const Array<T, N, Config2>& rhs )
{
   Array<T, N, Config1> cpy = lhs;
   cpy -= rhs;
   return cpy;
}

DECLARE_NAMESPACE_END
