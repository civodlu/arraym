
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

DECLARE_NAMESPACE_NLL

DECLARE_NAMESPACE_NLL_END

struct TestArrayDimIterator
{
   void test_columns()
   {
      test_columns_impl<Array_row_major<float, 2>>();
      test_columns_impl<Array_column_major<float, 2>>();
      test_columns_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_columns_impl()
   {
      array_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      int c = 0;
      for (auto col : columns(a))
      {
         TESTER_ASSERT(col.shape() == vector2ui(1, 2));
         TESTER_ASSERT(col(0, 0) == a(c, 0));
         TESTER_ASSERT(col(0, 1) == a(c, 1));
         ++c;
      }
      TESTER_ASSERT(c == 3);
   }

   void test_columns_const()
   {
      test_columns_const_impl<Array_row_major<float, 2>>();
      test_columns_const_impl<Array_column_major<float, 2>>();
      test_columns_const_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_columns_const_impl()
   {
      array_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      int c = 0;
      for (auto col : columns(a))
      {
         TESTER_ASSERT(col.shape() == vector2ui(1, 2));
         TESTER_ASSERT(col(0, 0) == a(c, 0));
         TESTER_ASSERT(col(0, 1) == a(c, 1));
         ++c;
      }
      TESTER_ASSERT(c == 3);
   }

   void test_rows()
   {
      test_rows_impl<Array_row_major<float, 2>>();
      test_rows_impl<Array_column_major<float, 2>>();
      test_rows_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_rows_impl()
   {
      array_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      int r = 0;
      for (auto row : rows(a))
      {
         TESTER_ASSERT(row.shape() == vector2ui(3, 1));
         TESTER_ASSERT(row(0, 0) == a(0, r));
         TESTER_ASSERT(row(1, 0) == a(1, r));
         TESTER_ASSERT(row(2, 0) == a(2, r));
         ++r;
      }
      TESTER_ASSERT(r == 2);
   }

   void test_rows_const()
   {
      test_rows_const_impl<Array_row_major<float, 2>>();
      test_rows_const_impl<Array_column_major<float, 2>>();
      test_rows_const_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class array_type>
   void test_rows_const_impl()
   {
      array_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      const array_type& a_const = a;

      int r = 0;
      for (auto row : rows(a_const))
      {
         TESTER_ASSERT(row.shape() == vector2ui(3, 1));
         TESTER_ASSERT(row(0, 0) == a(0, r));
         TESTER_ASSERT(row(1, 0) == a(1, r));
         TESTER_ASSERT(row(2, 0) == a(2, r));
         ++r;
      }
      TESTER_ASSERT(r == 2);
   }

   void test_matrix_rows()
   {
      using matrix_type = Matrix<float>;

      matrix_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      int r = 0;
      for (auto row : rows(a))
      {
         TESTER_ASSERT(row.shape() == vector2ui(1, 2));
         TESTER_ASSERT(row(0, 0) == a(r, 0));
         TESTER_ASSERT(row(0, 1) == a(r, 1));
         ++r;
      }
   }

   void test_matrix_rows_const()
   {
      using matrix_type = Matrix<float>;

      matrix_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      const matrix_type& a_const = a;

      int r = 0;
      for (auto row : rows(a_const))
      {
         TESTER_ASSERT(row.shape() == vector2ui(1, 2));
         TESTER_ASSERT(row(0, 0) == a(r, 0));
         TESTER_ASSERT(row(0, 1) == a(r, 1));
         ++r;
      }
   }

   void test_matrix_columns()
   {
      using matrix_type = Matrix<float>;

      matrix_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      int r = 0;
      for (auto column : columns(a))
      {
         TESTER_ASSERT(column.shape() == vector2ui(3, 1));
         TESTER_ASSERT(column(0, 0) == a(0, r));
         TESTER_ASSERT(column(1, 0) == a(1, r));
         TESTER_ASSERT(column(2, 0) == a(2, r));
         ++r;
      }
   }

   void test_matrix_columns_const()
   {
      using matrix_type = Matrix<float>;

      matrix_type a(3, 2);
      a = {1, 2, 3, 4, 5, 6};

      const matrix_type& a_const = a;

      int r = 0;
      for (auto column : columns(a_const))
      {
         TESTER_ASSERT(column.shape() == vector2ui(3, 1));
         TESTER_ASSERT(column(0, 0) == a(0, r));
         TESTER_ASSERT(column(1, 0) == a(1, r));
         TESTER_ASSERT(column(2, 0) == a(2, r));
         ++r;
      }
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
      array_type a(2, 2, 3);
      a = {1, 2,  3,  4,

           5, 6,  7,  8,

           9, 10, 11, 12};

      int s = 0;
      for (auto slice : slices(a))
      {
         TESTER_ASSERT(slice.shape() == vector3ui(2, 2, 1));
         TESTER_ASSERT(slice(0, 0, 0) == a(0, 0, s));
         TESTER_ASSERT(slice(0, 1, 0) == a(0, 1, s));
         TESTER_ASSERT(slice(1, 0, 0) == a(1, 0, s));
         TESTER_ASSERT(slice(1, 1, 0) == a(1, 1, s));
         ++s;
      }
      TESTER_ASSERT(s == 3);
   }

   void test_slices_const()
   {
      test_slices_const_impl<Array_row_major<float, 3>>();
      test_slices_const_impl<Array_column_major<float, 3>>();
      test_slices_const_impl<Array_row_major_multislice<float, 3>>();
   }

   template <class array_type>
   void test_slices_const_impl()
   {
      array_type a(2, 2, 3);
      a = {1, 2,  3,  4,

           5, 6,  7,  8,

           9, 10, 11, 12};

      int s = 0;
      for (auto slice : slices(a))
      {
         TESTER_ASSERT(slice.shape() == vector3ui(2, 2, 1));
         TESTER_ASSERT(slice(0, 0, 0) == a(0, 0, s));
         TESTER_ASSERT(slice(0, 1, 0) == a(0, 1, s));
         TESTER_ASSERT(slice(1, 0, 0) == a(1, 0, s));
         TESTER_ASSERT(slice(1, 1, 0) == a(1, 1, s));
         ++s;
      }
      TESTER_ASSERT(s == 3);
   }
   
   void test_values()
   {
      test_values_impl<Array_row_major<char, 2>>();
      test_values_impl<Array_column_major<int, 2>>();
      test_values_impl<Array_row_major_multislice<size_t, 2>>();
   }

   template <class array_type>
   void test_values_impl()
   {
      // test access & l-value
      array_type a(2, 3);
      a = { 0, 1, 2, 3, 4, 5};

      std::vector<int> count_values(6);

      int s = 0;
      for (auto& value : values(a))
      {   
         ++count_values[value];
         ++s;

         value += 2;
      }
      TESTER_ASSERT(s == a.size()); // iterated the number of array elements

      // we have a count of exactly 1 of each value
      TESTER_ASSERT(*std::min_element(count_values.begin(), count_values.end()) == 1);
      TESTER_ASSERT(*std::max_element(count_values.begin(), count_values.end()) == 1);

      TESTER_ASSERT(a(0, 0) == 2);
      TESTER_ASSERT(a(1, 0) == 3);
      TESTER_ASSERT(a(0, 1) == 4);
      TESTER_ASSERT(a(1, 1) == 5);
      TESTER_ASSERT(a(0, 2) == 6);
      TESTER_ASSERT(a(1, 2) == 7);
   }

   void test_values_const()
   {
      test_values_const_impl<Array_row_major<char, 2>>();
      test_values_const_impl<Array_column_major<int, 2>>();
      test_values_const_impl<Array_row_major_multislice<size_t, 2>>();
   }

   template <class array_type>
   void test_values_const_impl()
   {
      // test access & l-value
      array_type a(2, 3);
      a = { 0, 1, 2, 3, 4, 5 };

      const array_type& a_cpy = a;

      std::vector<int> count_values(6);

      int s = 0;
      for (auto& value : values(a_cpy))
      {
         ++count_values[value];
         ++s;
      }
      TESTER_ASSERT(s == a.size()); // iterated the number of array elements

      // we have a count of exactly 1 of each value
      TESTER_ASSERT(*std::min_element(count_values.begin(), count_values.end()) == 1);
      TESTER_ASSERT(*std::max_element(count_values.begin(), count_values.end()) == 1);
   }
};

TESTER_TEST_SUITE(TestArrayDimIterator);
TESTER_TEST(test_columns);
TESTER_TEST(test_columns_const);
TESTER_TEST(test_rows);
TESTER_TEST(test_rows_const);
TESTER_TEST(test_matrix_rows);
TESTER_TEST(test_matrix_rows_const);
TESTER_TEST(test_matrix_columns);
TESTER_TEST(test_matrix_columns_const);
TESTER_TEST(test_slices);
TESTER_TEST(test_slices_const);
TESTER_TEST(test_values);
TESTER_TEST(test_values_const);
TESTER_TEST_SUITE_END();
