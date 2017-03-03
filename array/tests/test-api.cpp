#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestArrayApi
{
   using vector_type = Array<float, 1>;
   using array_type = Array<float, 2>;
   using array_type_multislice = Array_row_major_multislice<float, 2>;
   using array_type_small = MatrixSmall_column_major<float, 16>;
   using matrix_type = Matrix<float>;

   void api_array_init()
   {
      array_type array(2, 3);

      // linearly initialize an array.
      // The array is filled by following the dimension order (x-axis, y-axis and so on) 
      // whatever the memory ordering (e.g., row major or column major)
      array = { 1, 2, 3, 4, 5, 6 };

      TESTER_ASSERT(array(0, 0) == 1);
      TESTER_ASSERT(array(1, 0) == 2);
      TESTER_ASSERT(array(0, 1) == 3);
      TESTER_ASSERT(array(1, 1) == 4);
      TESTER_ASSERT(array(0, 2) == 5);
      TESTER_ASSERT(array(1, 2) == 6);
   }

   void api_access()
   {
      array_type array(2, 3);

      // access the array by specifiying individual indexes
      array(0, 0) = 1;
      
      // access the array by specifiying a tuple individual
      array({ 1, 1 }) = 2;

      TESTER_ASSERT(array(0, 0) == 1);
      TESTER_ASSERT(array(1, 1) == 2);
   }

   void api_value_based_semantic()
   {
      array_type array1(2, 3);
      array_type array2(1, 1);
      
      // the array2 current memory is released and the content of array1 is copied
      array2 = array1;

      TESTER_ASSERT(&array1(0, 0) != &array2(0, 0));
   }

   void api_array_ref_tuple()
   {
      array_type array({ 10, 10 }, -1.0f);
      // create a reference of the subarray(2, 2) to (5, 5) inclusive
      // the coordinates are specified as a pair of points <min, max>
      auto sub_array = array({ 2, 2 }, { 5, 5 });
      sub_array = 42;  // the referenced array will be updated

      TESTER_ASSERT(array(0, 0) == -1.0f);
      TESTER_ASSERT(array(2, 2) == 42);
      TESTER_ASSERT(array(5, 5) == 42);
      TESTER_ASSERT(array(6, 6) == -1.0f);
   }

   void api_array_ref_range()
   {
      array_type array({ 10, 10 }, -1.0f);
      // create a reference of the subarray(2, 2) to (5, 5) inclusive
      // the coordinates are specified as a range
      auto sub_array = array(R(2, 5), R(2, 5));
      sub_array = 42;  // the referenced array will be updated

      TESTER_ASSERT(array(0, 0) == -1.0f);
      TESTER_ASSERT(array(2, 2) == 42);
      TESTER_ASSERT(array(5, 5) == 42);
      TESTER_ASSERT(array(6, 6) == -1.0f);
   }

   void api_array_ref_range_relative()
   {
      array_type array({ 10, 10 }, -1.0f);
      // create a reference of the subarray(2, 2) to (9, 8) inclusive
      // the coordinates are specified as a range with negative indexes
      // -1 = end of the array in the current dimension
      // -2 = element before the end of the array in the current dimension
      auto sub_array = array(R(7, -1), R(2, -2));
      sub_array = 42;  // the referenced array will be updated

      TESTER_ASSERT(array(7, 1) == -1.0f);
      TESTER_ASSERT(array(7, 2) == 42);
      TESTER_ASSERT(array(9, 2) == 42);
      TESTER_ASSERT(array(7, 9) == -1.0f);
      TESTER_ASSERT(array(7, 8) == 42);
   }

   void api_array_ref_range_all()
   {
      array_type array({ 10, 10 }, -1.0f);

      // capture the entire array in a reference and update it
      array(rangeAll, rangeAll) = 42;  

      TESTER_ASSERT(array(0, 0) == 42);
      TESTER_ASSERT(array(9, 9) == 42);
   }

   void api_small_array()
   {
      array_type_small array(4, 4);
      array = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };

      // small array optimization with small buffer allocated on the stack
      // in this case, array_type_small has a buffer of 16 float allocated on
      // the stack. if the array is smaller, use the stack buffer, else
      // we must resort to heap based memory.
      TESTER_ASSERT((char*)&array(0, 0) - (char*)&array < sizeof(array_type_small));
   }

#ifdef WITH_CUDA
   void api_cuda_array()
   {
      // initialize the memory on the CPU
      Array_column_major<float, 1> cpu_memory(4);
      cpu_memory = { 1, 2, 3, 4 };

      // transfer to GPU and run calculations
      Array_cuda_column_major<float, 1> gpu_memory = cpu_memory;
      gpu_memory = cos(gpu_memory);

      // once all calculations are performed, get the result back on the CPU
      Array_column_major<float, 1> cpu_result = gpu_memory;
      for (size_t index : range(cpu_memory))
      {
         TESTER_ASSERT(fabs(cpu_result(index) - std::cos(cpu_memory(index))) < 1e-5f);
      }
   }
#endif

   void api_dense_linear_algebra()
   {
      // demonstrate basic linear algebra functions
      matrix_type array(2, 2);
      array = { -1, 4, 5, -8 };
      TESTER_ASSERT(norm2(array * inv(array) - identity<float>(2)) < 1e-4f);
   }

   void api_stacking()
   {
      vector_type vector1(3);
      vector1 = { 1, 2, 3 };

      vector_type vector2(3);
      vector2 = { 4, 5, 6 };

      // create a higher dimensional array from a set of arrays
      auto vertical_stacking = stack(vector1, vector2);
      std::cout << vertical_stacking << std::endl;
      TESTER_ASSERT(vertical_stacking.shape() == vector2ui(3, 2));
      TESTER_ASSERT(vertical_stacking(0, 0) == vector1(0));
      TESTER_ASSERT(vertical_stacking(0, 1) == vector2(0));
   }

   void api_slices()
   {
      array_type array(2, 3);
      array = { 1, 2, 3, 4, 5, 6 };
      
      // slice an array following a given dimension (x) passing by coordinate (0, 0)
      auto slice_x_01 = array.slice<0>({ 0, 1 });
      slice_x_01 = 42;

      TESTER_ASSERT(slice_x_01.shape() == vector1ui(3));
      TESTER_ASSERT(slice_x_01(0) == 42);
      TESTER_ASSERT(slice_x_01(1) == 42);
      TESTER_ASSERT(slice_x_01(2) == 42);
   }

   void api_op()
   {
      array_type array(2, 3);
      array = { 1, 2, 3, 4, 5, 6 };

      // example of basic functions on arrays
      const auto mean_value = mean(array);
      const auto max_value = max(array);
      const auto min_value = min(array);
      TESTER_ASSERT(mean_value == (1 + 2 + 3 + 4 + 5 + 6) / 6.0f);
      TESTER_ASSERT(max_value == 6);
      TESTER_ASSERT(min_value == 1);
   }

   void api_op_axis()
   {
      array_type array(3, 2);
      array = { 1, 2, 3, 4, 5, 6 };

      // example of basic functions on arrays following an axis
      // e.g., axis = x, create a new array of dim N - 1, with axis x aggregated
      // r = { 
      //       f(array(rangeAll, 0)),
      //       f(array(rangeAll, 1)),
      //       ...
      //       f(array(rangeAll, N)),
      //     }
      const auto mean_value = mean(array, 0);
      std::cout << mean_value << std::endl;
      const auto max_value = max(array, 0);
      std::cout << max_value << std::endl;
      const auto min_value = min(array, 0);
      std::cout << min_value << std::endl;

      TESTER_ASSERT(mean_value.shape() == vector1ui(2));
      TESTER_ASSERT(mean_value(0) == (1 + 2 + 3) / 3.0f);
      TESTER_ASSERT(mean_value(1) == (4 + 5 + 6) / 3.0f);
      TESTER_ASSERT(max_value.shape() == vector1ui(2));
      TESTER_ASSERT(max_value(0) == 3);
      TESTER_ASSERT(max_value(1) == 6);
      TESTER_ASSERT(min_value.shape() == vector1ui(2));
      TESTER_ASSERT(min_value(0) == 1);
      TESTER_ASSERT(min_value(1) == 4);
   }

   void api_repmat()
   {
      array_type array(3, 2);
      array = { 1, 2, 3, 4, 5, 6 };

      // Replicate an array in multiple dimensions
      auto r = repmat(array, vector3ui(1, 1, 2));
      TESTER_ASSERT(r.shape() == vector3ui(3, 2, 2));
      TESTER_ASSERT(r(0, 0, 0) == array(0, 0));
      TESTER_ASSERT(r(1, 0, 0) == array(1, 0));
      TESTER_ASSERT(r(2, 0, 0) == array(2, 0));

      TESTER_ASSERT(r(0, 1, 0) == array(0, 1));
      TESTER_ASSERT(r(1, 1, 0) == array(1, 1));
      TESTER_ASSERT(r(2, 1, 0) == array(2, 1));

      TESTER_ASSERT(r(0, 0, 1) == array(0, 0));
      TESTER_ASSERT(r(1, 0, 1) == array(1, 0));
      TESTER_ASSERT(r(2, 0, 1) == array(2, 0));

      TESTER_ASSERT(r(0, 1, 1) == array(0, 1));
      TESTER_ASSERT(r(1, 1, 1) == array(1, 1));
      TESTER_ASSERT(r(2, 1, 1) == array(2, 1));
   }

   void api_dense_linear_algebra_subblocks()
   {
      // demonstrate basic linear algebra functions
      Matrix_column_major<float> array(6, 7);

      // address only a 2x2 sub-block
      auto sub_2x2 = array(R(1, 2), R(2, 3));

      // initialize the sub-block in axis-order fashion
      sub_2x2 = { -1, 4, 5, -8 };

      // compute its inverse
      auto sub_2x2_inv = inv(sub_2x2);
      
      // verify inverse properties ||A * inv(A) - I||_2 == 0
      TESTER_ASSERT(norm2(sub_2x2 * sub_2x2_inv - identity<float>(2)) < 1e-4f);
   }

   template <class T, class Config, size_t N>
   void Test1(Array<T, N, Config>& lhs)
   {
   }

   void api_enumerate_rows()
   {
      array_type array(2, 3);
      array = { 1, 2,
                3, 4,
                5, 6 };

      for (auto it : enumerate(rows(array)))
      {
         auto ref_array = *it;
         ref_array += 1;
      }

      TESTER_ASSERT(array(0, 0) == 2);
      TESTER_ASSERT(array(0, 1) == 4);
      TESTER_ASSERT(array(0, 2) == 6);
      TESTER_ASSERT(array(1, 0) == 3);
      TESTER_ASSERT(array(1, 1) == 5);
      TESTER_ASSERT(array(1, 2) == 7);
   }

   void api_enumerate_vectors()
   {
      std::vector<int> v = { 1, 2, 3, 4 };
      size_t index = 0;
      for (auto it : enumerate(v))
      {
         *it *= 2;
         ++index;
      }

      TESTER_ASSERT(index == 4);
      TESTER_ASSERT(v[0] == 2);
      TESTER_ASSERT(v[1] == 4);
      TESTER_ASSERT(v[2] == 6);
      TESTER_ASSERT(v[3] == 8);
   }
};

//
// http://pages.cs.wisc.edu/~cs701-1/LectureNotes/trunk/cs701-lec-12-1-2015/cs701-lec-12-01-2015.pdf
//


TESTER_TEST_SUITE(TestArrayApi);
TESTER_TEST(api_enumerate_vectors);
TESTER_TEST(api_enumerate_rows);
TESTER_TEST(api_array_init);
TESTER_TEST(api_access);
TESTER_TEST(api_value_based_semantic);
TESTER_TEST(api_array_ref_tuple);
TESTER_TEST(api_array_ref_range);
TESTER_TEST(api_array_ref_range_relative);
TESTER_TEST(api_array_ref_range_all);
TESTER_TEST(api_small_array);
#ifdef WITH_CUDA
TESTER_TEST(api_cuda_array);
#endif
TESTER_TEST(api_dense_linear_algebra);
TESTER_TEST(api_stacking);
TESTER_TEST(api_slices);
TESTER_TEST(api_op);
TESTER_TEST(api_op_axis);
TESTER_TEST(api_repmat);
TESTER_TEST(api_dense_linear_algebra_subblocks);
TESTER_TEST_SUITE_END();