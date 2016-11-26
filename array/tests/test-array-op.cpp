#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayOp
{
   void test_matrixAdd()
   {
      test_matrixAdd_impl<NAMESPACE_NLL::Array<int, 2>>();  // naive, contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array<float, 2>>();  // BLAS, contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();  // naive, non fully contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>();  // BLAS, non fully contiguous
   }

   template <class Array>
   void test_matrixAdd_impl()
   {
      
      Array a1( 2, 3 );
      a1 = {
         1, 2, 3, 4, 5, 6
      };

      Array a2( 2, 3 );
      a2 = {
         11, 12, 13, 14, 15, 16
      };

      Array result = a1;
      NAMESPACE_NLL::details::array_add( result, a2 );

      TESTER_ASSERT( result( 0, 0 ) == a1( 0, 0 ) + a2( 0, 0 ) );
      TESTER_ASSERT( result( 1, 0 ) == a1( 1, 0 ) + a2( 1, 0 ) );

      TESTER_ASSERT( result( 0, 1 ) == a1( 0, 1 ) + a2( 0, 1 ) );
      TESTER_ASSERT( result( 1, 1 ) == a1( 1, 1 ) + a2( 1, 1 ) );

      TESTER_ASSERT( result( 0, 2 ) == a1( 0, 2 ) + a2( 0, 2 ) );
      TESTER_ASSERT( result( 1, 2 ) == a1( 1, 2 ) + a2( 1, 2 ) );
   }
};

TESTER_TEST_SUITE( TestArrayOp );
TESTER_TEST( test_matrixAdd );
TESTER_TEST_SUITE_END();