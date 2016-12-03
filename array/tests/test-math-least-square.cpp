#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestLeastSquare
{
   template <class Array>
   void random(Array& array)
   {
      auto f = [](typename Array::value_type)
      {
         return generateUniformDistribution<float>(-5, 5);
      };

      auto op = [&](typename Array::value_type* y_pointer, ui32 y_stride, ui32 nb_elements)
      {
         NAMESPACE_NLL::details::apply_naive1(y_pointer, y_stride, nb_elements, f);
      };


      iterate_array(array, op);
   }

   void test_norm2()
   {
      test_norm2_impl<Array<float, 2>>();
      test_norm2_impl<Array<int, 2>>();
      test_norm2_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class T>
   T sqr( T v )
   {
      return v * v;
   }

   template <class Array>
   void test_norm2_impl()
   {
      Array a1( 2, 3 );
      a1 = { 1, 2, 3, 4, 5, 6 };

      const auto n = norm2( a1 );
      const double expected = std::sqrt(
         sqr( a1( 0, 0 ) ) +
         sqr( a1( 0, 1 ) ) +
         sqr( a1( 0, 2 ) ) +
         sqr( a1( 1, 0 ) ) +
         sqr( a1( 1, 1 ) ) +
         sqr( a1( 1, 2 ) ) );

      TESTER_ASSERT( fabs( expected - n ) < 1e-4 );
      std::cout << n << std::endl;
   }

   void test_simple()
   {
      
      for (unsigned int n = 0; n < 10000; ++n)
      {
         srand(n);
         const auto nb = generateUniformDistribution<size_t>(10, 20);
         const auto dim = generateUniformDistribution<size_t>(3, 10);
         const auto bdim = generateUniformDistribution<size_t>(1, 5);

         Matrix<float> a(vector2ui(nb, dim ));
         Matrix<float> x(vector2ui(dim, bdim ));

         random(a);
         random(x);
         
         const auto b = a * x;
         /*
         auto x_found = least_square(a, b);
         const auto b_ax = b - a * x_found;

         std::cout << b_ax << std::endl;
         const double residual = norm2(b_ax);
         TESTER_ASSERT(residual < 1e-4f);*/
      }
   }
};

TESTER_TEST_SUITE(TestLeastSquare);
TESTER_TEST( test_norm2 );
TESTER_TEST(test_simple);
TESTER_TEST_SUITE_END();
