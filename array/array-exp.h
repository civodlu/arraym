#pragma once

DECLARE_NAMESPACE_NLL

//
// TODO only for contiguous arrays...
//

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

   operator value_type() const
   {
      return value;
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
   using result_type = typename A::result_type;


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
};

template <class A, class B, class Op>
class ExprBinOp 
{
   const A& _left;
   const B& _right;

public:
   using left_type = A;
   using right_type = B;
   using operator_type = Op;
   using result_type = decltype( operator_type::apply( A(), B() ) );
      
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

   auto operator()() const -> decltype(operator_type::apply(_left, _right))
   {
      return operator_type::apply(_left, _right);
   }
};

class OpAdd;
class OpSub;
class OpMul;


template <class T, int N, class Config>
using ArrayArrayAdd = Expr < ExprBinOp<Array<T, N, Config>, Array<T, N, Config>, OpAdd> >;

class OpAdd
{
public:
   template <class T, int N, class Config>
   static Array<T, N, Config> apply( const Array<T, N, Config>& lhs, const Array<T, N, Config>& rhs )
   {
      // TODO handle sub-array! need a processor
      NLL_FAST_ASSERT( lhs.shape() == rhs.shape(), "must have the same shape!" );
      Array<T, N, Config> r( lhs );
      blas::axpy<T>( static_cast<blas::BlasInt>(lhs.size()), 1, &rhs( 0 ), 1, &r( 0 ), 1 );
      return r;
   }
};

template <class T, int N, class Config>
Expr<ExprBinOp<Array<T, N, Config>, Array<T, N, Config>, OpAdd> >
operator+( const Array<T, N, Config>& a, const Array<T, N, Config>& b )
{
   using ExprT = ExprBinOp<Array<T, N, Config>, Array<T, N, Config>, OpAdd>;
   return Expr<ExprT>( ExprT( a, b ) );
}

// https://github.com/guyz/cpp-array/blob/master/array/expr.hpp
// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Expression-template


DECLARE_NAMESPACE_END