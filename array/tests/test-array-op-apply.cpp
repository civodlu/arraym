#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

template <class T>
T f2x(T value)
{
   return value * 2;
}

struct TestArrayOpApply
{
   void test_array_apply_function()
   {
      test_array_apply_function_impl<Array<float, 2>>();
      test_array_apply_function_impl<Array_column_major<float, 2>>();
      test_array_apply_function_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_function_impl<Array<int, 2>>();
      test_array_apply_function_impl<Array_column_major<int, 2>>();
      test_array_apply_function_impl<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_array_apply_function_impl()
   {
      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };

      const auto a1_fun = constarray_apply_function(a1, f2x<typename Array::value_type>);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = static_cast<typename Array::value_type>(a1(x, y) * 2);
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   void test_array_apply_functions()
   {
      test_array_apply_functions_cos_impl<Array<float, 2>>();
      test_array_apply_functions_cos_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_sin_impl<Array<float, 2>>();
      test_array_apply_functions_sin_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_sqrt_impl<Array<float, 2>>();
      test_array_apply_functions_sqrt_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_abs_impl<Array<float, 2>>();
      test_array_apply_functions_abs_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_min_impl<Array<float, 2>>();
      test_array_apply_functions_min_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_max_impl<Array<float, 2>>();
      test_array_apply_functions_max_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_log_impl<Array<float, 2>>();
      test_array_apply_functions_log_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_exp_impl<Array<float, 2>>();
      test_array_apply_functions_exp_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_mean_impl<Array<float, 2>>();
      test_array_apply_functions_mean_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_sqr_impl<Array<float, 2>>();
      test_array_apply_functions_sqr_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class Array>
   void test_array_apply_functions_cos_impl()
   {
      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = cos(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::cos(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sin_impl()
   {
      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = sin(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::sin(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sqrt_impl()
   {
      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = sqrt(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::sqrt(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_abs_impl()
   {
      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = abs(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::abs(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sqr_impl()
   {
      Array a1( 2, 3 );
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = sqr( a1 );
      for ( size_t y = 0; y < a1.shape()[ 1 ]; ++y )
      {
         for ( size_t x = 0; x < a1.shape()[ 0 ]; ++x )
         {
            const auto expected = a1( x, y ) * a1( x, y );
            const auto found = a1_fun( x, y );
            TESTER_ASSERT( std::abs( expected - found ) < 1e-4f );
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_log_impl()
   {
      Array a1( 2, 3 );
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = log( a1 );
      for ( size_t y = 0; y < a1.shape()[ 1 ]; ++y )
      {
         for ( size_t x = 0; x < a1.shape()[ 0 ]; ++x )
         {
            const auto expected = std::log( a1( x, y ) );
            const auto found = a1_fun( x, y );
            TESTER_ASSERT( std::abs( expected - found ) < 1e-4f );
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_mean_impl()
   {
      Array a1( 2, 3 );
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = mean( a1 );
      for ( size_t y = 0; y < a1.shape()[ 1 ]; ++y )
      {
         for ( size_t x = 0; x < a1.shape()[ 0 ]; ++x )
         {
            const auto expected = (1+2+3+4+5+6) / (typename Array::value_type)(6);
            const auto found = a1_fun;
            TESTER_ASSERT( std::abs( expected - found ) < 1e-4f );
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_exp_impl()
   {
      Array a1( 2, 3 );
      a1 = { 1, 2, 3, 4, 5, 6 };
      const auto a1_fun = exp( a1 );
      for ( size_t y = 0; y < a1.shape()[ 1 ]; ++y )
      {
         for ( size_t x = 0; x < a1.shape()[ 0 ]; ++x )
         {
            const auto expected = std::exp( a1( x, y ) );
            const auto found = a1_fun( x, y );
            TESTER_ASSERT( std::abs( expected - found ) < 1e-4f );
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_min_impl()
   {
      Array a1(2, 3);
      a1 = { 2, 1, 3, 4, 5, 6 };
      const typename Array::value_type a1_fun = min(a1);
      TESTER_ASSERT(a1_fun == 1);
   }

   template <class Array>
   void test_array_apply_functions_max_impl()
   {
      Array a1(2, 3);
      a1 = { 2, 1, 3, 4, 5, 6 };
      const typename Array::value_type a1_fun = max(a1);
      TESTER_ASSERT(a1_fun == 6);
   }
};

TESTER_TEST_SUITE(TestArrayOpApply);
TESTER_TEST(test_array_apply_function);
TESTER_TEST(test_array_apply_functions);
TESTER_TEST_SUITE_END();
