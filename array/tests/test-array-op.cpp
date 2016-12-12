#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using vector3ui = StaticVector<ui32, 3>;
using vector2ui = StaticVector<ui32, 2>;
using vector1ui = StaticVector<ui32, 1>;

struct TestArrayOp
{
   void test_is_fully_contiguous()
   {
      static_assert(IsArrayLayoutContiguous<Array_row_major<int, 2>>::value, "row major is contiguous layout!");
      static_assert(IsArrayLayoutContiguous<Array_column_major<int, 2>>::value, "column major is contiguous layout!");
      static_assert(!IsArrayLayoutContiguous<Array_row_major_multislice<int, 2>>::value, "slice based memory is NOT contiguous layout!");

      static_assert(IsArrayLayoutLinear<Array_row_major<int, 2>>::value, "row major is contiguous layout!");
      static_assert(IsArrayLayoutLinear<Array_column_major<int, 2>>::value, "column major is contiguous layout!");
      static_assert(IsArrayLayoutLinear<Array_row_major_multislice<int, 2>>::value, "slice based memory is contiguous layout by slice!");

      {
         // trick for the slice based array, we share the implementation
         using array_type = Array_row_major_multislice<int, 3>;
         auto array       = array_type(4, 6, 10);
         TESTER_ASSERT(details::IsMemoryFullyContiguous<array_type::Memory>::value(array.getMemory(), std::integral_constant<bool, true>()));
      }

      {
         // trick for the slice based array, we share the implementation
         using array_type = Array_row_major_multislice<int, 3>;
         auto array       = array_type(10, 6, 4);
         TESTER_ASSERT(details::IsMemoryFullyContiguous<array_type::Memory>::value(array.getMemory(), std::integral_constant<bool, true>()));
      }

      TESTER_ASSERT(!is_array_fully_contiguous(Array_row_major_multislice<int, 2>(4, 6))); // multi-slices: not contiguous!

      TESTER_ASSERT(is_array_fully_contiguous(Array_row_major<int, 2>(4, 6)));       // contiguous layout, not a sub-array
      TESTER_ASSERT(is_array_fully_contiguous(Array_row_major<int, 3>(6, 4, 1)));    // contiguous layout, not a sub-array
      TESTER_ASSERT(is_array_fully_contiguous(Array_column_major<int, 3>(6, 4, 2))); // contiguous layout, not a sub-array

      TESTER_ASSERT(is_array_fully_contiguous(Array_column_major<int, 3>(6, 4, 1)));
      TESTER_ASSERT(is_array_fully_contiguous(Array_column_major<int, 3>(6, 1, 4)));
      TESTER_ASSERT(is_array_fully_contiguous(Array_row_major<int, 3>(6, 1, 4)));
      TESTER_ASSERT(is_array_fully_contiguous(Array_row_major<int, 3>(6, 4, 1)));
      TESTER_ASSERT(is_array_fully_contiguous(Array_row_major<int, 3>(1, 4, 2)));
      TESTER_ASSERT(is_array_fully_contiguous(Array_column_major<int, 3>(1, 1, 4)));

      Array_row_major<int, 3> a1(10, 11, 12);
      TESTER_ASSERT(!is_array_fully_contiguous(a1(vector3ui(1, 1, 1), vector3ui(3, 3, 3))));
      TESTER_ASSERT(is_array_fully_contiguous(a1(vector3ui(0, 0, 5), vector3ui(9, 10, 8))));
   }

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

      test_matrixSubOp_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      test_matrixSubOp_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      test_matrixSubOp_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      test_matrixSubOp_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous
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
      testArray_mul_cte_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      testArray_mul_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      testArray_mul_cte_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_mul_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_mul_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_mul_cte_left_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous
   }

   template <class Array>
   void testArray_mul_cte_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      Array cpy = a1;

      cpy *= static_cast<typename Array::value_type>(2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(1, 0) == a1(1, 0) * 2);

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

      cpy = a1 * static_cast<typename Array::value_type>(2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(1, 0) == a1(1, 0) * 2);

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

      cpy = static_cast<typename Array::value_type>(2) * a1;
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) * 2);
      TESTER_ASSERT(cpy(1, 0) == a1(1, 0) * 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) * 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) * 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) * 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) * 2);
   }

   void testArray_div_cte()
   {
      testArray_div_cte_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      testArray_div_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      testArray_div_cte_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_div_cte_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous

      testArray_div_cte_right_impl<NAMESPACE_NLL::Array<int, 2>>();                        // naive, contiguous
      testArray_div_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 2>>();   // naive, non fully contiguous
      testArray_div_cte_right_impl<NAMESPACE_NLL::Array<float, 2>>();                      // BLAS, contiguous
      testArray_div_cte_right_impl<NAMESPACE_NLL::Array_row_major_multislice<float, 2>>(); // BLAS, non fully contiguous
   }

   template <class Array>
   void testArray_div_cte_impl()
   {
      Array a1(2, 3);
      a1 = {11, 21, 31, 40, 50, 60};

      Array cpy = a1;

      cpy /= static_cast<typename Array::value_type>(2);
      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) / 2);
      TESTER_ASSERT(cpy(1, 0) == a1(1, 0) / 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) / 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) / 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) / 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) / 2);
   }

   template <class Array>
   void testArray_div_cte_right_impl()
   {
      Array a1(2, 3);
      a1 = {11, 21, 31, 40, 50, 60};

      Array cpy = a1 / static_cast<typename Array::value_type>(2);

      TESTER_ASSERT(cpy(0, 0) == a1(0, 0) / 2);
      TESTER_ASSERT(cpy(1, 0) == a1(1, 0) / 2);

      TESTER_ASSERT(cpy(0, 1) == a1(0, 1) / 2);
      TESTER_ASSERT(cpy(1, 1) == a1(1, 1) / 2);

      TESTER_ASSERT(cpy(0, 2) == a1(0, 2) / 2);
      TESTER_ASSERT(cpy(1, 2) == a1(1, 2) / 2);
   }

   void testArray_mul_array()
   {
      testArray_mul_array<NAMESPACE_NLL::Matrix_row_major<int>>();      // naive, contiguous
      testArray_mul_array<NAMESPACE_NLL::Matrix_column_major<int>>();   // naive, contiguous
      testArray_mul_array<NAMESPACE_NLL::Matrix_row_major<float>>();    // BLAS, contiguous
      testArray_mul_array<NAMESPACE_NLL::Matrix_column_major<float>>(); // BLAS, contiguous
   }

   template <class Array>
   void testArray_mul_array()
   {
      Array a1(2, 3);
      a1 = {2, 3, 4, 5, 6, 7};

      Array a2(3, 2);
      a2 = {20, 30, 40, 50, 60, 70};


      Array result = a1 * a2;
      TESTER_ASSERT(result.shape() == vector2ui(2, 2));

      TESTER_ASSERT(result(0, 0) == 400);
      TESTER_ASSERT(result(1, 0) == 490);
      TESTER_ASSERT(result(0, 1) == 760);
      TESTER_ASSERT(result(1, 1) == 940);
   }

   void testArray_matrix_memoryOrder()
   {
      {
         Matrix_row_major<int> m1(4, 10);
         auto order = getMatrixMemoryOrder(m1);
         TESTER_ASSERT(order == CBLAS_ORDER::CblasRowMajor);
      }

      {
         Matrix_row_major<int> m1(10, 4);
         auto order = getMatrixMemoryOrder(m1);
         TESTER_ASSERT(order == CBLAS_ORDER::CblasRowMajor);
      }

      {
         Matrix_column_major<int> m1(4, 10);
         auto order = getMatrixMemoryOrder(m1);
         TESTER_ASSERT(order == CBLAS_ORDER::CblasColMajor);
      }

      {
         Matrix_column_major<int> m1(10, 4);
         auto order = getMatrixMemoryOrder(m1);
         TESTER_ASSERT(order == CBLAS_ORDER::CblasColMajor);
      }

      {
         Array<int, 2> m1(10, 4); // not a matrix, just an array!
         auto order = getMatrixMemoryOrder(m1);
         TESTER_ASSERT(order == CBLAS_ORDER::UnkwownMajor);
      }
   }

   void testMatrix_transpose()
   {
      testMatrix_transpose_impl<NAMESPACE_NLL::Matrix_row_major<int>>();      // naive, contiguous
      testMatrix_transpose_impl<NAMESPACE_NLL::Matrix_column_major<int>>();   // naive, contiguous
      testMatrix_transpose_impl<NAMESPACE_NLL::Matrix_row_major<float>>();    // BLAS, contiguous
      testMatrix_transpose_impl<NAMESPACE_NLL::Matrix_column_major<float>>(); // BLAS, contiguous
   }

   template <class Array>
   void testMatrix_transpose_impl()
   {
      Array m1(3, 2);
      m1             = {1, 2, 3, 4, 5, 6};
      const auto m1t = transpose(m1);

      TESTER_ASSERT(m1t.shape() == vector2ui(2, 3));
      TESTER_ASSERT(m1t(0, 0) == 1);
      TESTER_ASSERT(m1t(0, 1) == 2);
      TESTER_ASSERT(m1t(0, 2) == 3);

      TESTER_ASSERT(m1t(1, 0) == 4);
      TESTER_ASSERT(m1t(1, 1) == 5);
      TESTER_ASSERT(m1t(1, 2) == 6);
   }
};

TESTER_TEST_SUITE(TestArrayOp);
TESTER_TEST(test_same_data_ordering);
TESTER_TEST(test_is_fully_contiguous);
TESTER_TEST(test_matrixAdd);
TESTER_TEST(test_matrixSub);
TESTER_TEST(testArray_mul_cte);
TESTER_TEST(testArray_div_cte);
TESTER_TEST(testArray_mul_array);
TESTER_TEST(testArray_matrix_memoryOrder);
TESTER_TEST(testMatrix_transpose);
TESTER_TEST_SUITE_END();