#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayExp
{
   void test_expr_add_array_array_impl()
   {
      test_expr_add_array_array_impl<Array<float, 2>>();
   }
   template <class Array>
   void test_expr_add_array_array_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array a3(2, 3);
      a3 = {110, 120, 130, 140, 150, 160};
      /*
      auto expr = a1 + a2;
      Array result = a3 + expr;

      TESTER_ASSERT( result( 0, 0 ) == a1( 0, 0 ) + a2( 0, 0 ) + a3( 0, 0 ) );
      TESTER_ASSERT( result( 1, 0 ) == a1( 1, 0 ) + a2( 1, 0 ) + a3( 1, 0 ) );

      TESTER_ASSERT( result( 0, 1 ) == a1( 0, 1 ) + a2( 0, 1 ) + a3( 0, 1 ) );
      TESTER_ASSERT( result( 1, 1 ) == a1( 1, 1 ) + a2( 1, 1 ) + a3( 1, 1 ) );

      TESTER_ASSERT( result( 0, 2 ) == a1( 0, 2 ) + a2( 0, 2 ) + a3( 0, 2 ) );
      TESTER_ASSERT( result( 1, 2 ) == a1( 1, 2 ) + a2( 1, 2 ) + a3( 1, 2 ) );*/
   }
};

TESTER_TEST_SUITE(TestArrayExp);
TESTER_TEST(test_expr_add_array_array_impl);
TESTER_TEST_SUITE_END();