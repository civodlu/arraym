#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestMatrixGemm
{
   void test_gemm_columnMajor()
   {
      test_gemm_columnMajorImpl<Matrix_column_major<float>>();
      test_gemm_columnMajorImpl<Matrix_column_major<double>>();
   }

   template <class matrix_type>
   void test_gemm_columnMajorImpl()
   {
      // http://www.ibm.com/support/knowledgecenter/SSFHY8_5.5.0/com.ibm.cluster.essl.v5r5.essl100.doc/am5gr_hsgemm.htm
      using value_type  = typename matrix_type::value_type;
      using memory_type = typename matrix_type::Memory;

      value_type A_values[] = {1.0, 2.0, -1.0, -1.0, 4.0,  2.0,  0.0,  1.0, 1.0,  -1.0, 1.0, -1.0, -1.0, 1.0, 2.0, -3.0, 2.0, 2.0, 2.0, 0.0,
                               4.0, 0.0, -2.0, 1.0,  -1.0, -1.0, -1.0, 1.0, -3.0, 2.0,  999, 999,  999,  999, 999, 999,  999, 999, 999, 999};

      //
      // Trick the ordering: here we have row memory ordering to be fited to
      // column ordering
      //
      matrix_type A(memory_type(vector2ui(6, 5), A_values, vector2ui(5, 1)));

      value_type B_values[] = {1.0, -1.0, 0.0, 2.0, 2.0, 2.0, -1.0, -2.0, 1.0, 0.0, -1.0, 1.0, -3.0, -1.0, 1.0, -1.0, 4.0, 2.0, -1.0, 1.0, 999, 999, 999, 999};

      matrix_type B(memory_type(vector2ui(5, 4), B_values, vector2ui(4, 1)));

      value_type C_values[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 999, 999, 999, 999};

      matrix_type C(memory_type(vector2ui(6, 4), C_values, vector2ui(4, 1)));

      NAMESPACE_NLL::details::gemm<value_type>(1, A, B, 2, C);

      matrix_type C_expected(6, 4);
      C_expected = {
          24.0, 13.0, -5.0, 3.0, -3.0, -4.0, 2.0, 4.0, 4.0, 1.0, 2.0, 5.0, -2.0, 6.0, -1.0, -9.0, -4.0, -6.0, 5.0, 5.0, 16.0, 7.0, -4.0, 7.0,
      };

      const auto diff = C - C_expected;

      const auto error = norm2(C - C_expected);

      TESTER_ASSERT(error < 1e-4);
   }

   void test_processor_different_ordering_dataRowMajor()
   {
      test_processor_different_ordering_impl<Array_column_major<float, 2>>();
      test_processor_different_ordering_impl<Array_column_major<double, 2>>();
      test_processor_different_ordering_impl<Array_column_major<char, 2>>();

      test_processor_different_ordering_impl<Array_row_major<float, 2>>();
      test_processor_different_ordering_impl<Array_row_major<double, 2>>();
      test_processor_different_ordering_impl<Array_row_major<char, 2>>();
   }

   template <class array_type>
   void test_processor_different_ordering_impl()
   {
      using value_type  = typename array_type::value_type;
      using memory_type = typename array_type::Memory;

      value_type values[] = {1, 2, 3, 4, 5, 6};

      // make sure we have the expected array
      array_type array(memory_type(vector2ui(3, 2), values, vector2ui(1, 3)));
      TESTER_ASSERT(array(0, 0) == 1);
      TESTER_ASSERT(array(1, 0) == 2);
      TESTER_ASSERT(array(2, 0) == 3);
      TESTER_ASSERT(array(0, 1) == 4);
      TESTER_ASSERT(array(1, 1) == 5);
      TESTER_ASSERT(array(2, 1) == 6);

      // now copy the array to another with a different memory ordering
      array_type array2 = array;
      TESTER_ASSERT(&array2(0, 0) != &array(0, 0));

      // make sure we have copied correctly despite the memory ordering
      TESTER_ASSERT(array2(0, 0) == 1);
      TESTER_ASSERT(array2(1, 0) == 2);
      TESTER_ASSERT(array2(2, 0) == 3);
      TESTER_ASSERT(array2(0, 1) == 4);
      TESTER_ASSERT(array2(1, 1) == 5);
      TESTER_ASSERT(array2(2, 1) == 6);
   }

   void test_processor_different_ordering_dataColumnMajor()
   {
      test_processor_different_orderingCol_impl<Array_column_major<float, 2>>();
      test_processor_different_orderingCol_impl<Array_column_major<double, 2>>();
      test_processor_different_orderingCol_impl<Array_column_major<char, 2>>();

      test_processor_different_orderingCol_impl<Array_row_major<float, 2>>();
      test_processor_different_orderingCol_impl<Array_row_major<double, 2>>();
      test_processor_different_orderingCol_impl<Array_row_major<char, 2>>();
   }

   template <class array_type>
   void test_processor_different_orderingCol_impl()
   {

      using value_type  = typename array_type::value_type;
      using memory_type = typename array_type::Memory;

      value_type values[] = {1, 2, 3, 4, 5, 6};

      // make sure we have the expected array
      array_type array(memory_type(vector2ui(2, 3), values, vector2ui(1, 2)));
      TESTER_ASSERT(array(0, 0) == 1);
      TESTER_ASSERT(array(1, 0) == 2);
      TESTER_ASSERT(array(0, 1) == 3);
      TESTER_ASSERT(array(1, 1) == 4);
      TESTER_ASSERT(array(0, 2) == 5);
      TESTER_ASSERT(array(1, 2) == 6);

      // now copy the array to another with a different memory ordering
      array_type array2 = array;
      TESTER_ASSERT(&array2(0, 0) != &array(0, 0));

      // make sure we have copied correctly despite the memory ordering
      TESTER_ASSERT(array2(0, 0) == 1);
      TESTER_ASSERT(array2(1, 0) == 2);
      TESTER_ASSERT(array2(0, 1) == 3);
      TESTER_ASSERT(array2(1, 1) == 4);
      TESTER_ASSERT(array2(0, 2) == 5);
      TESTER_ASSERT(array2(1, 2) == 6);
   }

   void test_gemm_inv()
   {
      test_gemm_invImpl<Matrix_column_major<float>>();
      test_gemm_invImpl<Matrix_column_major<double>>();
   }

   template <class Matrix>
   void test_gemm_invImpl()
   {
      using value_type  = typename Matrix::value_type;
      using memory_type = typename Matrix::Memory;
      using matrix_type = Matrix;

      value_type A_values[] = {1, 2, 1, 999, -3, 4, -1, 999};

      matrix_type A(memory_type(vector2ui(3, 2), A_values, vector2ui(1, 4)));

      value_type B_values[] = {1, 2, 1, -3, 4, -1};

      matrix_type B(memory_type(vector2ui(3, 2), B_values, vector2ui(1, 3)));

      value_type C_values[] = {
          0.5, 0.5, 0.5, 999, 999, 0.5, 0.5, 0.5, 999, 999, 0.5, 0.5, 0.5, 999, 999,
      };

      matrix_type C(memory_type(vector2ui(3, 3), C_values, vector2ui(1, 5)));

      // see example 2, http://www.ibm.com/support/knowledgecenter/SSFHY8_5.5.0/com.ibm.cluster.essl.v5r5.essl100.doc/am5gr_hsgemm.htm
      NAMESPACE_NLL::details::gemm<value_type>(false, true, 1, A, B, 2, C);

      TESTER_ASSERT(C(0, 0) == 11);
      TESTER_ASSERT(C(1, 0) == -9);
      TESTER_ASSERT(C(2, 0) == 5);

      TESTER_ASSERT(C(0, 1) == -9);
      TESTER_ASSERT(C(1, 1) == 21);
      TESTER_ASSERT(C(2, 1) == -1);

      TESTER_ASSERT(C(0, 2) == 5);
      TESTER_ASSERT(C(1, 2) == -1);
      TESTER_ASSERT(C(2, 2) == 3);
   }
};

TESTER_TEST_SUITE(TestMatrixGemm);
TESTER_TEST(test_gemm_inv);
TESTER_TEST(test_gemm_columnMajor);
TESTER_TEST(test_processor_different_ordering_dataRowMajor);
TESTER_TEST(test_processor_different_ordering_dataColumnMajor);
TESTER_TEST_SUITE_END();
