#pragma once

/**
 @file

 This file implements expression templates and contrary to most other libraries, the purpose is not to
 avoid temporaries but to use as much as possible the BLAS interface. In particular, this code has been
 strongly inspired from : 
 "A C++ 11 implementation of arbitrary-rank tensors for high-performance computing", Alejandro M. Aragon
 https://github.com/guyz/cpp-array/tree/master/array

 Often we want to be able to use Array of the same rank and type but with a different Config parameter,
 in particular to enable custom-fixed memory allocator with general purpose heap allocators.
 */

#ifdef WITH_EXPRESSION_TEMPLATE

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

   ExprConstant() : _value(value)
   {
   }

   result_type operator()() const
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
   using left_type     = const A&;
   using right_type    = const B&;
   using operator_type = Op;
   //using result_type = decltype( operator_type::apply( *( (A*)( nullptr ) ), *( (B*)( nullptr ) ) ) );
   using result_type = decltype(operator_type::apply(A(), B()));

   ExprBinOp(const A& left, const B& right) : _left(left), _right(right)
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

   result_type operator()() const
   {
      return operator_type::apply(_left, _right);
   }
};

template <class A, class B, class Op>
class RefExprBinOp
{
   A& _left;
   const B& _right;

   //  using _result = decltype( operator_type::apply( A(), B() ) ) &;

public:
   using left_type     = A&;
   using right_type    = const B&;
   using operator_type = Op;
   using result_type   = decltype(operator_type::apply(A(), B()))&;

   RefExprBinOp(A& left, const B& right) : _left(left), _right(right)
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

   result_type operator()() const
   {
      return operator_type::apply(_left, _right);
   }
};

class OpAdd;
class OpSub;
class OpMul;

template <class T, int N, class Config, class Config2>
using ArrayArrayAdd = Expr<ExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>>;

template <class T, int N, class Config>
using ScalarArrayMul = Expr<ExprBinOp<ExprConstant<T>, Array<T, N, Config>, OpMul>>;

class OpAdd
{
public:
   /// Array + Array
   template <class T, int N, class Config, class Config2>
   static Array<T, N, Config> apply(const Array<T, N, Config>& lhs, const Array<T, N, Config2>& rhs)
   {
      Array<T, N, Config> cpy = lhs;
      array_add(cpy, rhs);
      return cpy;
   }

   /*
   template <class A, class B>
   static decltype(A() + B())
   apply( const Expr<A>& a, const Expr<B>& b )
   {
      return a() + b();
   }*/

   /// Array += Array
   template <class T, int N, class Config, class Config2>
   static Array<T, N, Config>& apply(Array<T, N, Config>& lhs, const Array<T, N, Config2>& rhs)
   {
      array_add(lhs, rhs);
      return lhs;
   }

   /*
   /// Array += Expr
   template <class T, int N, class Config, class B>
   static Array<T, N, Config>& apply(Array<T, N, Config>& lhs, const Expr<B>& b)
   {
      lhs += b();
      return lhs;
   }*/
};

class OpMul
{
public:
   // scalar * array
   template <typename T, size_t N, class Config>
   static Array<T, N, Config> apply(const ExprConstant<T>& a, const Array<T, N, Config>& b)
   {
      Array<T, N, Config> r = b;
      details::array_mul(r, a());
      return r;
   }

   // array * scalar
   template <typename T, size_t N, class Config>
   static Array<T, N, Config> apply(const Array<T, N, Config>& b, const ExprConstant<T>& a)
   {
      Array<T, N, Config> r = b;
      details::array_mul(r, a());
      return r;
   }

   // array *= scalar
   template <typename T, size_t N, class Config>
   static Array<T, N, Config>& apply(Array<T, N, Config>& b, const ExprConstant<T>& a)
   {
      details::array_mul(b, a());
      return b;
   }
};

//
// operator+ (Array, Array)
//
template <class T, int N, class Config, class Config2>
Expr<ExprBinOp<Array_TemplateExpressionEnabled<T, N, Config>, Array<T, N, Config2>, OpAdd>> operator+(const Array<T, N, Config>& a,
                                                                                                      const Array<T, N, Config2>& b)
{
   using ExprT = ExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>;
   return Expr<ExprT>(ExprT(a, b));
}
/*
// (Array, Expr)
template <class T, int N, class Config, class A>
Expr<ExprBinOp<Array_TemplateExpressionEnabled<T, N, Config>, Expr<A>, OpAdd>> operator+( const Array<T, N, Config>& a,
                                                                                          const Expr<A>& b )
{
   using ExprT = ExprBinOp<Array<T, N, Config>, Expr<A>, OpAdd> ;
   return Expr<ExprT>( ExprT( a, b ) );
}
*/
//
// operator+= (Array, Array)
//
template <class T, int N, class Config, class Config2>
Array_TemplateExpressionEnabled<T, N, Config>& operator+=(Array<T, N, Config>& a, const Array<T, N, Config2>& b)
{
   using ExprT = RefExprBinOp<Array<T, N, Config>, Array<T, N, Config2>, OpAdd>;
   return Expr<ExprT>(ExprT(a, b))();
}

//
// operator* (Array, scalar)
//
template <class T, int N, class Config>
Array_TemplateExpressionEnabled<T, N, Config> operator*(const Array<T, N, Config>& a, T value)
{
   using ExprT = typename ScalarArrayMul<T, N, Config>::expression_type;
   return ScalarArrayMul<T, N, Config>(ExprT(ExprConstant<T>(value), a));
}

template <class T, int N, class Config>
Array_TemplateExpressionEnabled<T, N, Config> operator*(T value, const Array<T, N, Config>& a)
{
   using ExprT = typename ScalarArrayMul<T, N, Config>::expression_type;
   return ScalarArrayMul<T, N, Config>(ExprT(ExprConstant<T>(value), a));
}

template <class T, int N, class Config>
Array_TemplateExpressionEnabled<T, N, Config>& operator*=(Array<T, N, Config>& a, T value)
{
   using ExprT             = RefExprBinOp<Array<T, N, Config>, ExprConstant<T>, OpMul>;
   using RefScalarArrayMul = Expr<ExprT>;

   return RefScalarArrayMul(ExprT(a, ExprConstant<T>(value)));
}

// https://github.com/guyz/cpp-array/blob/master/array/expr.hpp
// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Expression-template

// TODO reconsider boost.proto:
// http://web.archive.org/web/20120906070131/http://cpp-next.com/archive/2011/01/expressive-c-expression-optimization/

DECLARE_NAMESPACE_NLL_END

#endif
