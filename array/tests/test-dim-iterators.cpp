
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

DECLARE_NAMESPACE_NLL


DECLARE_NAMESPACE_NLL_END

struct TestArrayDimIterator
{
   void test_rows()
   {
      test_rows_impl<Array_row_major<float, 2>>();
      test_rows_impl<Array_column_major<float, 2>>();
      test_rows_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_rows_impl()
   {
      array_type a( 3, 2 );
      a = { 1, 2, 3,
         4, 5, 6 };

      int r = 0;
      for ( auto row : rows( a ) )
      {
         TESTER_ASSERT( row.shape() == vector2ui( 3, 1 ) );
         TESTER_ASSERT( row( 0, 0 ) == a( 0, r ) );
         TESTER_ASSERT( row( 1, 0 ) == a( 1, r ) );
         TESTER_ASSERT( row( 2, 0 ) == a( 2, r ) );
         ++r;
      }
      TESTER_ASSERT( r == 2 );
   }
   
   void test_const_rows()
   {
      test_const_rows_impl<Array_row_major<float, 2>>();
      test_const_rows_impl<Array_column_major<float, 2>>();
      test_const_rows_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_const_rows_impl()
   {
      array_type a( 3, 2 );
      a = { 1, 2, 3,
         4, 5, 6 };

      const auto& a_const = a;

      int r = 0;

      auto row_it = rows( a_const );

      
      for ( auto row : rows( a_const ) )
      {
         TESTER_ASSERT( row.shape() == vector2ui( 3, 1 ) );
         TESTER_ASSERT( row( 0, 0 ) == a( 0, r ) );
         TESTER_ASSERT( row( 1, 0 ) == a( 1, r ) );
         TESTER_ASSERT( row( 2, 0 ) == a( 2, r ) );
         ++r;
      }
      TESTER_ASSERT( r == 2 );
   }

   void test_columns()
   {
      test_columns_impl<Array_row_major<float, 2>>();
      test_columns_impl<Array_column_major<float, 2>>();
      test_columns_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_columns_impl()
   {
      array_type a( 3, 2 );
      a = { 1, 2, 3,
         4, 5, 6 };

      int r = 0;
      for ( auto col : columns( a ) )
      {
         TESTER_ASSERT( col.shape() == vector2ui( 1, 2 ) );
         TESTER_ASSERT( col( 0, 0 ) == a( r, 0 ) );
         TESTER_ASSERT( col( 0, 1 ) == a( r, 1 ) );
         ++r;
      }
      TESTER_ASSERT( r == 3 );
   }

   void test_const_columns()
   {
      test_const_columns_impl<Array_row_major<float, 2>>();
      test_const_columns_impl<Array_column_major<float, 2>>();
      test_const_columns_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_const_columns_impl()
   {
      array_type a( 3, 2 );
      a = { 1, 2, 3,
         4, 5, 6 };

      const array_type& a_const = a;

      int r = 0;
      for ( auto col : columns( a_const ) )
      {
         TESTER_ASSERT( col.shape() == vector2ui( 1, 2 ) );
         TESTER_ASSERT( col( 0, 0 ) == a( r, 0 ) );
         TESTER_ASSERT( col( 0, 1 ) == a( r, 1 ) );
         ++r;
      }
      TESTER_ASSERT( r == 3 );
   }

   void test_const_slices()
   {
      test_const_slices_impl<Array_row_major<float, 3>>();
      test_const_slices_impl<Array_column_major<float, 3>>();
      test_const_slices_impl<Array_row_major_multislice<float, 3>>();
   }

   template <class array_type>
   void test_const_slices_impl()
   {
      array_type a( 3, 2, 2 );
      a = { 1, 2, 3,
         4, 5, 6,
         11, 12, 13,
         14, 15, 16 };

      const array_type& a_const = a;

      int r = 0;
      for ( auto s : slices( a_const ) )
      {
         TESTER_ASSERT( s.shape() == vector3ui( 3, 2, 1 ) );
         TESTER_ASSERT( s( 0, 0, 0 ) == a( 0, 0, r ) );
         TESTER_ASSERT( s( 1, 0, 0 ) == a( 1, 0, r ) );
         TESTER_ASSERT( s( 2, 0, 0 ) == a( 2, 0, r ) );

         TESTER_ASSERT( s( 0, 1, 0 ) == a( 0, 1, r ) );
         TESTER_ASSERT( s( 1, 1, 0 ) == a( 1, 1, r ) );
         TESTER_ASSERT( s( 2, 1, 0 ) == a( 2, 1, r ) );
         ++r;
      }
      TESTER_ASSERT( r == 2 );
   }

   void test_slices()
   {
      test_slices_impl<Array_row_major<float, 3>>();
      test_slices_impl<Array_column_major<float, 3>>();
      test_slices_impl<Array_row_major_multislice<float, 3>>();
   }

   template <class array_type>
   void test_slices_impl()
   {
      array_type a( 3, 2, 2 );
      a = { 1, 2, 3,
         4, 5, 6,
         11, 12, 13,
         14, 15, 16 };

      int r = 0;
      for ( auto s : slices( a ) )
      {
         TESTER_ASSERT( s.shape() == vector3ui( 3, 2, 1 ) );
         TESTER_ASSERT( s( 0, 0, 0 ) == a( 0, 0, r ) );
         TESTER_ASSERT( s( 1, 0, 0 ) == a( 1, 0, r ) );
         TESTER_ASSERT( s( 2, 0, 0 ) == a( 2, 0, r ) );

         TESTER_ASSERT( s( 0, 1, 0 ) == a( 0, 1, r ) );
         TESTER_ASSERT( s( 1, 1, 0 ) == a( 1, 1, r ) );
         TESTER_ASSERT( s( 2, 1, 0 ) == a( 2, 1, r ) );
         ++r;
      }
      TESTER_ASSERT( r == 2 );
   }
};

TESTER_TEST_SUITE(TestArrayDimIterator);
TESTER_TEST(test_rows);
TESTER_TEST( test_const_rows );
TESTER_TEST( test_columns );
TESTER_TEST( test_const_columns );
TESTER_TEST( test_const_slices );
TESTER_TEST( test_slices );
TESTER_TEST_SUITE_END();