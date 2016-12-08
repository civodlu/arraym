#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestMatrixInv
{
   template <class Array>
   void random( Array& array )
   {
      auto f = []( typename Array::value_type )
      {
         return generateUniformDistribution<float>( -5, 5 );
      };

      auto op = [&]( typename Array::value_type* y_pointer, ui32 y_stride, ui32 nb_elements )
      {
         NAMESPACE_NLL::details::apply_naive1( y_pointer, y_stride, nb_elements, f );
      };


      iterate_array( array, op );
   }

   void test_random()
   {
      test_random_impl<Matrix_row_major<float>>();
      test_random_impl<Matrix_column_major<float>>();
      test_random_impl<Matrix_row_major<double>>();
      test_random_impl<Matrix_column_major<double>>();
   }

   template <class Matrix>
   void test_random_impl()
   {
      for ( unsigned int n = 0; n < 1000; ++n )
      {
         srand( n );
         const auto nb = generateUniformDistribution<size_t>( 3, 10 );

         Matrix id( vector2ui( nb, nb ) );
         identity( id );

         Matrix a( vector2ui( nb, nb ) );
         random( a );

         const Matrix b = inv( a );

         {
            const auto i2 = b * a - id;
            const double residual = norm2( i2 );
            if( residual > 1e-2 )
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT( residual < 1e-2f );
         }

         {
            const auto i2 = a * b - id;
            const double residual = norm2( i2 );
            if ( residual > 1e-2 )
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT( residual < 1e-2f );
         }
      }
   }

   void test_random_subarray()
   {
      test_random_subarray_impl<Matrix_row_major<float>>();
   }

   template <class Matrix>
   void test_random_subarray_impl()
   {
      for (unsigned int n = 0; n < 1000; ++n)
      {
         srand(n);
         const auto dim_x = generateUniformDistribution<size_t>(40, 50);
         const auto dim_y = generateUniformDistribution<size_t>(40, 50);
         const auto origin_x = generateUniformDistribution<size_t>(0, 8);
         const auto origin_y = generateUniformDistribution<size_t>(0, 15);
         const auto nb = generateUniformDistribution<size_t>(3, 15);

         Matrix source({ dim_y, dim_x });
         random(source);

         Matrix id(vector2ui(nb, nb));
         identity(id);

         auto a = source({ origin_y, origin_x }, { origin_y + nb - 1, origin_x + nb - 1 });
         const Matrix b = inv(a);

         {
            const auto i2 = b * a - id;
            const double residual = norm2(i2);
            if (residual > 1e-2)
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT(residual < 1e-2f);
         }

         {
            const auto i2 = a * b - id;
            const double residual = norm2(i2);
            if (residual > 1e-2)
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT(residual < 1e-2f);
         }
      }
   }
};

TESTER_TEST_SUITE( TestMatrixInv );
TESTER_TEST(test_random_subarray);
TESTER_TEST(test_random);
TESTER_TEST_SUITE_END();
