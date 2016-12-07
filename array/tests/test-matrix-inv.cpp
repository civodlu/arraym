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

   void test_simple()
   {
      for ( unsigned int n = 0; n < 10000; ++n )
      {
         srand( n );
         const auto nb = generateUniformDistribution<size_t>( 3, 10 );


         Matrix<float> a( vector2ui( nb, nb ) );
         random( a );

         const Matrix<float> b = inv( a );

         {
            const auto i2 = b * a - identity<float>( nb );
            const double residual = norm2( i2 );
            if( residual > 1e-2 )
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT( residual < 1e-2f );
         }

         {
            const auto i2 = a * b - identity<float>( nb );
            const double residual = norm2( i2 );
            if ( residual > 1e-2 )
            {
               std::cout << i2 << std::endl;
            }
            TESTER_ASSERT( residual < 1e-2f );
         }
      }
   }


};

TESTER_TEST_SUITE( TestMatrixInv );
TESTER_TEST( test_simple );
TESTER_TEST_SUITE_END();
