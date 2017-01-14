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
{
};

template <class T>
struct is_same<T> : public std::true_type
{
};

namespace details
{
   struct TypelistEmpty{};
}

template <class... args>
struct first;

template <class T1, class... Tx>
struct first<T1, Tx...>
{
   using type = T1;
};

template <>
struct first<>
{
   using type = details::TypelistEmpty;
};

template <typename... Args>
struct tuple_split
{
};

template <typename First, typename... Tail>
struct tuple_split<std::tuple<First, Tail...>>
{
   using head = First;
   using tail = std::tuple<Tail...>;
};

template <typename T>
struct remove_cvr
{
   using type = typename std::remove_cv<typename std::remove_reference<T>::type>::type;
};

/**
@brief Detect the number of arguments, the argument type and return type of function, member function, lamda
This implementation is based on http://stackoverflow.com/questions/22630832/get-argument-type-of-template-callable-object
http://functionalcpp.wordpress.com/2013/08/05/function-traits/
*/
template <class F>
struct function_traits;

// function pointer
template <class R, class... Args>
struct function_traits<R (*)(Args...)> : public function_traits<R(Args...)>
{
};

template <class R, class... Args>
struct function_traits<R(Args...)>
{
   using return_type = R;

   static const size_t arity = sizeof...(Args);

   using arguments = std::tuple<Args...>;

   template <std::size_t N>
   struct argument
   {
      static_assert(N < arity, "error: invalid parameter index.");
      using type = typename std::tuple_element<N, arguments>::type;
   };
};

// member function pointer
template <class C, class R, class... Args>
struct function_traits<R (C::*)(Args...)> : public function_traits<R(C&, Args...)>
{
};

// const member function pointer
template <class C, class R, class... Args>
struct function_traits<R (C::*)(Args...) const> : public function_traits<R(C&, Args...)>
{
};

// member object pointer
template <class C, class R>
struct function_traits<R(C::*)> : public function_traits<R(C&)>
{
};

// functor
template <class F>
struct function_traits
{
private:
   using call_type = function_traits<decltype(&F::operator())>;

public:
   using return_type = typename call_type::return_type;

   using arguments = typename tuple_split<typename call_type::arguments>::tail;

   enum
   {
      arity = call_type::arity - 1
   };

   template <std::size_t N>
   struct argument
   {
      static_assert(N < arity, "error: invalid parameter index.");
      using type = typename call_type::template argument<N + 1>::type;
   };
};

template <class F>
struct function_traits<F&> : public function_traits<F>
{
};

template <class F>
struct function_traits<F&&> : public function_traits<F>
{
};

namespace details
{
struct can_call_test
{
   template <typename F, typename... A>
   static decltype(std::declval<F>()(std::declval<A>()...), std::true_type()) f(int);

   template <typename F, typename... A>
   static std::false_type f(...);
};
}

/**
@brief return true if the function can be called with the provided parameters
*/
template <typename Function, typename... Params>
struct is_callable_with
{
   static const bool value = std::is_same<std::true_type, decltype(details::can_call_test::f<Function, Params...>(0))>::value;
};

template <class T>
struct PromoteFloating
{
   using type = float;
};

template <>
struct PromoteFloating<double>
{
   using type = double;
};



/**
@brief check the types provided are all identical
*/
template <class... args>
struct is_same_nocvr;

template <class x1, class x2, class... args>
struct is_same_nocvr<x1, x2, args...>
{
   static const bool value = std::is_same<typename remove_cvr<x1>::type, typename remove_cvr<x2>::type>::value &&
      is_same_nocvr<typename remove_cvr<x2>::type, args...>::value;
};

template <>
struct is_same_nocvr<> : public std::true_type
{
};

template <class T>
struct is_same_nocvr<T> : public std::true_type
{
};



DECLARE_NAMESPACE_NLL_END