#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayOp
{
   void test_same_data_ordering()
   {
      TESTER_ASSERT(same_data_ordering(Array<int, 2>(2, 2), Array<int, 2>(2, 2)));
      TESTER_ASSERT(!same_data_ordering(Array_row_major<int, 2>(2, 2), Array_column_major<int, 2>(2, 2)));

      TESTER_ASSERT(same_data_ordering(Array_row_major<int, 2>(), Array_column_major<int, 2>())); // no memory allocated, should be the same ordering!
   }

   void test_matrixAdd()
   {
      test_matrixAdd_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      test_matrixAdd_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      test_matrixAddOp_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      test_matrixAddOp_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      test_matrixAddOp_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      test_matrixAddOp_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      test_matrixAddOpInPlace_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      test_matrixAddOpInPlace_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      test_matrixAddOpInPlace_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      test_matrixAddOpInPlace_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous
   }

   template <class Array>
   void test_matrixAdd_impl()
   {

      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array result = a1;
      NAMESPACE_NLL::details::array_add(result, a2);

      TESTER_ASSERT(result(0, 0) == a1(0, 0) + a2(0, 0));
      TESTER_ASSERT(result(1, 0) == a1(1, 0) + a2(1, 0));

      TESTER_ASSERT(result(0, 1) == a1(0, 1) + a2(0, 1));
      TESTER_ASSERT(result(1, 1) == a1(1, 1) + a2(1, 1));

      TESTER_ASSERT(result(0, 2) == a1(0, 2) + a2(0, 2));
      TESTER_ASSERT(result(1, 2) == a1(1, 2) + a2(1, 2));
   }

   template <class Array>
   void test_matrixAddOp_impl()
   {

      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array result = a1 + a2;

      TESTER_ASSERT(result(0, 0) == a1(0, 0) + a2(0, 0));
      TESTER_ASSERT(result(1, 0) == a1(1, 0) + a2(1, 0));

      TESTER_ASSERT(result(0, 1) == a1(0, 1) + a2(0, 1));
      TESTER_ASSERT(result(1, 1) == a1(1, 1) + a2(1, 1));

      TESTER_ASSERT(result(0, 2) == a1(0, 2) + a2(0, 2));
      TESTER_ASSERT(result(1, 2) == a1(1, 2) + a2(1, 2));
   }

   template <class Array>
   void test_matrixAddOpInPlace_impl()
   {

      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array result = a1;
      result += a2;

      TESTER_ASSERT(result(0, 0) == a1(0, 0) + a2(0, 0));
      TESTER_ASSERT(result(1, 0) == a1(1, 0) + a2(1, 0));

      TESTER_ASSERT(result(0, 1) == a1(0, 1) + a2(0, 1));
      TESTER_ASSERT(result(1, 1) == a1(1, 1) + a2(1, 1));

      TESTER_ASSERT(result(0, 2) == a1(0, 2) + a2(0, 2));
      TESTER_ASSERT(result(1, 2) == a1(1, 2) + a2(1, 2));
   }

   void test_matrixSub()
   {
      test_matrixSub_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      test_matrixSub_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      test_matrixSub_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      test_matrixSub_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      // implemented but commented!
      //test_matrixSubOp_impl<NAMESPACE_NLL::Array<int, 2>>();   // naive, contiguous
      //test_matrixSubOp_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();  // naive, non fully contiguous
      /*
      // not implemented!
      test_matrixSubOp_impl<NAMESPACE_NLL::Array<float, 2>>(); // BLAS, contiguous
      test_matrixSubOp_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>();  // BLAS, non fully contiguous
      */
   }

   template <class Array>
   void test_matrixSub_impl()
   {

      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array result = a1;
      NAMESPACE_NLL::details::array_sub(result, a2);

      TESTER_ASSERT(result(0, 0) == a1(0, 0) - a2(0, 0));
      TESTER_ASSERT(result(1, 0) == a1(1, 0) - a2(1, 0));

      TESTER_ASSERT(result(0, 1) == a1(0, 1) - a2(0, 1));
      TESTER_ASSERT(result(1, 1) == a1(1, 1) - a2(1, 1));

      TESTER_ASSERT(result(0, 2) == a1(0, 2) - a2(0, 2));
      TESTER_ASSERT(result(1, 2) == a1(1, 2) - a2(1, 2));
   }

   template <class Array>
   void test_matrixSubOp_impl()
   {

      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array a2(2, 3);
      a2 = {11, 12, 13, 14, 15, 16};

      Array result = a1 - a2;

      TESTER_ASSERT(result(0, 0) == a1(0, 0) - a2(0, 0));
      TESTER_ASSERT(result(1, 0) == a1(1, 0) - a2(1, 0));

      TESTER_ASSERT(result(0, 1) == a1(0, 1) - a2(0, 1));
      TESTER_ASSERT(result(1, 1) == a1(1, 1) - a2(1, 1));

      TESTER_ASSERT(result(0, 2) == a1(0, 2) - a2(0, 2));
      TESTER_ASSERT(result(1, 2) == a1(1, 2) - a2(1, 2));
   }

   void testArray_mul_cte()
   {
      testArray_mul_cte_impl<NAMESPACE_NLL::Array<int, 2>>();                      // naive, contiguous
      testArray_mul_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>(); // naive, non fully contiguous
      // TODO
      //testArray_mul_cte_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      //testArray_mul_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array<int, 2>>();                      // naive, contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>(); // naive, non fully contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array<int, 2>>();                      // naive, contiguous
      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>(); // naive, non fully contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous
   }

   template <class Array>
   void testArray_mul_cte_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array cpy = a1;

      cpy *= static_cast<Array::value_type>(2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) * 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) * 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) * 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) * 2);
   }

   template <class Array>
   void testArray_mul_cte_right_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array cpy;

      cpy = a1 * static_cast<Array::value_type>(2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) * 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) * 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) * 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) * 2);
   }

   template <class Array>
   void testArray_mul_cte_left_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array cpy;

      cpy = static_cast<Array::value_type>(2) * a1;
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) * 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) * 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) * 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) * 2);
   }
};

TESTER_TEST_SUITE(TestArrayOp);
TESTER_TEST(test_same_data_ordering);
TESTER_TEST(test_matrixAdd);
TESTER_TEST(test_matrixSub);
TESTER_TEST(testArray_mul_cte);
TESTER_TEST_SUITE_END();