#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Simplify the std::enable_if expression so that it is readable
*/
template <class T, int N, class Config>
using Array_TemplateExpressionEnabled = typename std::enable_if<!array_use_naive_operator<Array<T, N, Config>>::value, Array<T, N, Config>>::type;

template <typename T>
class ExprConstant
{
public:
   using value_type  = T;
   using result_type = T;

   ExprConstant(value_type value) : _value(value)
   {
   }

   template <typename... Args>
   value_type operator()(Args... params) const
   {
      return _value;
   }

   operator value_type() const
   {
      return _value;
   }

private:
   T _value;
};

template <class A>
class Expr
{
   A _a;

public:
   using expression_type = A;
   using result_type     = typename A::result_type;

   Expr(const A& x) : _a(x)
   {
   }

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
};

template <class A, class B, class Op>
class ExprBinOp
{
   const A& _left;
   const B& _right;

public:
   using left_type     = A;
   using right_type    = B;
   using operator_type = Op;
   using result_type   = decltype(operator_type::apply(A(), B()));

   ExprBinOp(const A& left, const B& right) : _left(left), _right(right)
   {
   }

   auto left() const -> decltype(_left)
   {
      return _left;
   }

   auto right() const -> decltype(_right)
   {
      return _right;
   }

   auto operator()() const -> decltype(operator_type::apply(_left, _right))
   {
      return operator_type::apply(_left, _right);
   }
};

template <class A, class B, class Op>
class RefExprBinOp
{
   A& _left;
   const B& _right;

public:
   using left_type = A&;
   using right_type = const B&;
   using operator_type = Op;
   using result_type = decltype( operator_type::apply( A(), B() ) );

   RefExprBinOp( A& left, const B& right ) : _left( left ), _right( right )
   {
   }

   left_type left() const
   {
      return _left;
   }

   right_type right() const
   {
      return _right;
   }

   auto operator()() const -> decltype( operator_type::apply( _left, _right ) )
   {
      return operator_type::apply( _left, _right );
   }
};

class OpAdd;
class OpSub;
class OpMul;

//template <class T, int N, class Config, class Config2>
//using ArrayArrayAdd = Expr<ExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>>;

class OpAdd
{
public:
   template <class T, int N, class Config, class Config2>
   static Array<T, N, Config> apply(const Array<T, N, Config>& lhs, const Array<T, N, Config2>& rhs)
   {
      Array<T, N, Config> cpy = lhs;
      array_add(cpy, rhs);
      return cpy;
   }
   
   template <class T, int N, class Config, class Config2>
   static Array<T, N, Config>& apply( Array<T, N, Config>& lhs, const Array<T, N, Config2>& rhs )
   {
      array_add( lhs, rhs );
      return lhs;
   }
};


//
// operator+
//
template <class T, int N, class Config, class Config2>
Expr<ExprBinOp<Array_TemplateExpressionEnabled<T, N, Config>, Array<T, N, Config2>, OpAdd >>
operator+( const Array<T, N, Config>& a, const Array<T, N, Config2>& b )
{
   using ExprT = ExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>;
   return Expr<ExprT>(ExprT(a, b));
}

//
// operator+=
//
template <class T, int N, class Config, class Config2>
Array_TemplateExpressionEnabled<T, N, Config>&
operator+=( Array<T, N, Config>& a, const Array<T, N, Config2>& b )
{
   using ExprT = RefExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>;
   return Expr<ExprT>( ExprT( a, b ) )();
}

// https://github.com/guyz/cpp-array/blob/master/array/expr.hpp
// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Expression-template

DECLARE_NAMESPACE_END