#pragma once

DECLARE_NAMESPACE_NLL

template <typename... Params>
struct ReturnType;


template <typename T>
class ExprConstant
{
public:
   using value_type = T;
   using result_type = T;

   ExprConstant(value_type value) : _value(value)
   {}

   template <typename... Args>
   value_type operator()(Args... params) const
   {
      return x;
   }

private:
   T _value;
};

template <class A>
class Expr {
public:
   using expression_type = A;
   using result_type = typename A::result_type;

   Expr() : _a()
   {}

   Expr(const A& x) :_a(x)
   {}

   auto left() const -> decltype(_a.left())
   {
      return _a.left();
   }

   auto right() const -> decltype(_a.right())
   {
      return _a.right();
   }

   operator result_type()
   {
      return _a();
   }

   result_type operator()() const
   {
      return _a();
   }

private:
   A _a;
};

template <class A, class B, class Op>
class ExprBinOp {
public:
   using left_type = A;
   using right_type = B;
   using operator_type = Op;
   using result_type = typename ReturnType<left_type, right_type, operator_type>::result_type;

   ExprBinOp(const A& left, const B& right) : _left(left), _right(right)
   {}

   auto left() const -> decltype(_left)
   {
      return _left;
   }

   auto right() const -> decltype(_right)
   {
      return _right;
   }

   auto operator()() const -> decltype(operator_type::apply(a_, b_))
   {
      return operator_type::apply(_left, _right);
   }

private:
   const A& _left;
   const B& _right;
};

// https://github.com/guyz/cpp-array/blob/master/array/expr.hpp
// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Expression-template


DECLARE_NAMESPACE_END