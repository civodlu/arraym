#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayLogicalOp
{
   void test_lessthan()
   {
      test_lessthan_impl<Array<float, 2>>();
      test_lessthan_impl<Array_column_major<float, 2>>();
      test_lessthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
                 5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
                 50, -10, 9 };

      auto r = array1 < array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_lessthan_value()
   {
      test_lessthan_value_impl<Array<float, 2>>();
      test_lessthan_value_impl<Array_column_major<float, 2>>();
      test_lessthan_value_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessthan_value_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 3 };

      auto r = array1 < 4;
      auto r2 = 4 < array1;

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 1);
      TESTER_ASSERT(norm2(r2 - !r) == 0);
   }

   void test_greaterthan_value()
   {
      test_greaterthan_value_impl<Array<float, 2>>();
      test_greaterthan_value_impl<Array_column_major<float, 2>>();
      test_greaterthan_value_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterthan_value_impl()
   {
      array_type array1( 3, 2 );
      array1 = { 2, 6, 3,
         5, -1, 3 };

      auto r = array1 > 4;
      auto r2 = 4 > array1;

      TESTER_ASSERT( r( 0, 0 ) == 0 );
      TESTER_ASSERT( r( 1, 0 ) == 1 );
      TESTER_ASSERT( r( 2, 0 ) == 0 );
      TESTER_ASSERT( r( 0, 1 ) == 1 );
      TESTER_ASSERT( r( 1, 1 ) == 0 );
      TESTER_ASSERT( r( 2, 1 ) == 0 );
      TESTER_ASSERT( norm2( r2 - !r ) == 0 );
   }

   void test_lessequalthan_value()
   {
      test_lessequalthan_value_impl<Array<float, 2>>();
      test_lessequalthan_value_impl<Array_column_major<float, 2>>();
      test_lessequalthan_value_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessequalthan_value_impl()
   {
      array_type array1( 3, 2 );
      array1 = { 2, 6, 3,
                 5, -1, 4 };

      auto r = array1 <= 4;
      auto r2 = 4 <= array1;

      TESTER_ASSERT( r( 0, 0 ) == 1 );
      TESTER_ASSERT( r( 1, 0 ) == 0 );
      TESTER_ASSERT( r( 2, 0 ) == 1 );
      TESTER_ASSERT( r( 0, 1 ) == 0 );
      TESTER_ASSERT( r( 1, 1 ) == 1 );
      TESTER_ASSERT( r( 2, 1 ) == 1 );
      
      TESTER_ASSERT( r2( 0, 0 ) == 0 );
      TESTER_ASSERT( r2( 1, 0 ) == 1 );
      TESTER_ASSERT( r2( 2, 0 ) == 0 );
      TESTER_ASSERT( r2( 0, 1 ) == 1 );
      TESTER_ASSERT( r2( 1, 1 ) == 0 );
      TESTER_ASSERT( r2( 2, 1 ) == 1 );
   }

   void test_greaterequalthan_value()
   {
      test_greaterequalthan_value_impl<Array<float, 2>>();
      test_greaterequalthan_value_impl<Array_column_major<float, 2>>();
      test_greaterequalthan_value_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterequalthan_value_impl()
   {
      array_type array1( 3, 2 );
      array1 = { 2, 6, 3,
         5, -1, 4 };

      auto r = array1 >= 4;
      auto r2 = 4 >= array1;

      TESTER_ASSERT( r( 0, 0 ) == 0 );
      TESTER_ASSERT( r( 1, 0 ) == 1 );
      TESTER_ASSERT( r( 2, 0 ) == 0 );
      TESTER_ASSERT( r( 0, 1 ) == 1 );
      TESTER_ASSERT( r( 1, 1 ) == 0 );
      TESTER_ASSERT( r( 2, 1 ) == 1 );

      TESTER_ASSERT( r2( 0, 0 ) == 1 );
      TESTER_ASSERT( r2( 1, 0 ) == 0 );
      TESTER_ASSERT( r2( 2, 0 ) == 1 );
      TESTER_ASSERT( r2( 0, 1 ) == 0 );
      TESTER_ASSERT( r2( 1, 1 ) == 1 );
      TESTER_ASSERT( r2( 2, 1 ) == 1 );
   }

   void test_greaterthan()
   {
      test_greaterthan_impl<Array<float, 2>>();
      test_greaterthan_impl<Array_column_major<float, 2>>();
      test_greaterthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 > array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_greaterequalthan()
   {
      test_greaterequalthan_impl<Array<float, 2>>();
      test_greaterequalthan_impl<Array_column_major<float, 2>>();
      test_greaterequalthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_greaterequalthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 >= array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_lessequalthan()
   {
      test_lessequalthan_impl<Array<float, 2>>();
      test_lessequalthan_impl<Array_column_major<float, 2>>();
      test_lessequalthan_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_lessequalthan_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 9 };

      auto r = array1 <= array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_and()
   {
      test_and_impl<Array<char, 2>>();
      test_and_impl<Array_column_major<int, 2>>();
      test_and_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_and_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
                 1, 1, 0 };

      array_type array2(3, 2);
      array2 = { 1, 0, 1,
                 0, 1, 0 };

      auto r = array1 & array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_or()
   {
      test_or_impl<Array<char, 2>>();
      test_or_impl<Array_column_major<int, 2>>();
      test_or_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_or_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
         1, 1, 0 };

      array_type array2(3, 2);
      array2 = { 1, 0, 1,
         0, 1, 0 };

      auto r = array1 | array2;
      Array<int, 2> r2(r);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 1);
      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);
   }

   void test_not()
   {
      test_not_impl<Array<char, 2>>();
      test_not_impl<Array_column_major<int, 2>>();
      test_not_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_not_impl()
   {
      array_type array1(3, 2);
      array1 = { 0, 1, 1,
                 1, 1, 0 };

      auto r = !array1;
      
      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 0);
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);
   }

   void test_equal()
   {
      test_equal_impl<Array<float, 2>>();
      test_equal_impl<Array_column_major<float, 2>>();
      test_equal_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_equal_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 4 };

      auto r = equal_elementwise(array1, array2);
      
      TESTER_ASSERT(r(0, 1) == 0);
      TESTER_ASSERT(r(1, 1) == 0);
      TESTER_ASSERT(r(2, 1) == 1);

      TESTER_ASSERT(r(0, 0) == 0);
      TESTER_ASSERT(r(1, 0) == 1);
      TESTER_ASSERT(r(2, 0) == 0);
   }

   template <class array_type>
   void test_notequal_impl()
   {
      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };

      array_type array2(3, 2);
      array2 = { 1, 6, 8,
         50, -10, 4 };

      auto r = different_elementwise(array1, array2);

      TESTER_ASSERT(r(0, 1) == 1);
      TESTER_ASSERT(r(1, 1) == 1);
      TESTER_ASSERT(r(2, 1) == 0);

      TESTER_ASSERT(r(0, 0) == 1);
      TESTER_ASSERT(r(1, 0) == 0);
      TESTER_ASSERT(r(2, 0) == 1);
   }

   void test_count()
   {
      using array_type = Array_column_major<int, 2>;

      array_type array1( 3, 2 );
      array1 = { 2, 6, 3,
                 5, -1, 4 };
      auto r = count( array1, []( int v )
      {
         return v >= 4;
      } );

      TESTER_ASSERT( r == 3 );
   }

   void test_count_axis()
   {
      using array_type = Array_column_major<int, 2>;

      array_type array1(3, 2);
      array1 = { 2, 6, 3,
         5, -1, 4 };
      
      auto r = count(array1, 0, [](int v)
      {
         return v >= 4;
      });

      TESTER_ASSERT(r.shape() == vector1ui(2));
      TESTER_ASSERT(r(0) == 1);
      TESTER_ASSERT(r(1) == 2);
   }
};

TESTER_TEST_SUITE(TestArrayLogicalOp);
TESTER_TEST(test_lessthan_value);
TESTER_TEST(test_lessequalthan_value);
TESTER_TEST(test_greaterthan_value);
TESTER_TEST(test_greaterequalthan_value);
TESTER_TEST(test_lessthan);
TESTER_TEST(test_lessequalthan);
TESTER_TEST(test_greaterthan);
TESTER_TEST(test_greaterequalthan);
TESTER_TEST(test_lessequalthan);
TESTER_TEST(test_equal);
TESTER_TEST(test_or);
TESTER_TEST(test_and);
TESTER_TEST(test_not);
TESTER_TEST(test_count);
TESTER_TEST(test_count_axis);
TESTER_TEST_SUITE_END();