#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief check the types provided are all identical
*/
template <class... args>
struct is_same;

template <class x1, class x2, class... args>
struct is_same<x1, x2, args...>
{
   static const bool value = std::is_same<x1, x2>::value && is_same<x2, args...>::value;
};

template <>
struct is_same<> : public std::true_type
{};

template <class T>
struct is_same<T> : public std::true_type
{};

template <class T1, class ...Tx>
struct first
{
   using type = T1;
};

template <typename T>
struct remove_cvr
{
   using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};
DECLARE_NAMESPACE_END