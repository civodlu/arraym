#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestArrayRepmat
{
   void testRepmat_simple()
   {
      testRepmat_simple_impl<Array<int, 1>>();
      testRepmat_simple_impl<Array<float, 1>>();
#ifdef WITH_CUDA
      testRepmat_simple_impl<Array_cuda_column_major<float, 1>>();
#endif
   }

   template <class Array>
   void testRepmat_simple_impl()
   {
      Array vec(3);
      vec = {1, 2, 3};

      auto vec32 = repmat(vec, vector2ui(1, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(3, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);
   }

   void testRepmat_simple2()
   {
      Array<int, 1> vec(3);
      vec = {1, 2, 3};

      auto vec32 = repmat(vec, vector2ui(2, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(6, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);

      TESTER_ASSERT(vec32(0 + 3, 0) == 1);
      TESTER_ASSERT(vec32(1 + 3, 0) == 2);
      TESTER_ASSERT(vec32(2 + 3, 0) == 3);
      TESTER_ASSERT(vec32(0 + 3, 1) == 1);
      TESTER_ASSERT(vec32(1 + 3, 1) == 2);
      TESTER_ASSERT(vec32(2 + 3, 1) == 3);
   }

   void testRepmat_simple3()
   {
      Array<int, 1> vec(3);
      vec = {1, 2, 3};

      auto vec32 = repmat(vec, vector3ui(1, 2, 2));
      TESTER_ASSERT(vec32.shape() == vector3ui(3, 2, 2));
      TESTER_ASSERT(vec32(0, 0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0, 0) == 2);
      TESTER_ASSERT(vec32(2, 0, 0) == 3);
      TESTER_ASSERT(vec32(0, 1, 0) == 1);
      TESTER_ASSERT(vec32(1, 1, 0) == 2);
      TESTER_ASSERT(vec32(2, 1, 0) == 3);

      TESTER_ASSERT(vec32(0, 0, 1) == 1);
      TESTER_ASSERT(vec32(1, 0, 1) == 2);
      TESTER_ASSERT(vec32(2, 0, 1) == 3);
      TESTER_ASSERT(vec32(0, 1, 1) == 1);
      TESTER_ASSERT(vec32(1, 1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1, 1) == 3);
   }

   void testRepmat_strided()
   {
      int values[] = {1, 0, 2, 0, 3, 0};

      using array_type = Array<int, 1>;
      array_type vec(array_type::Memory(vector1ui(3), values, vector1ui(2)));
      TESTER_ASSERT(vec(0) == 1);
      TESTER_ASSERT(vec(1) == 2);
      TESTER_ASSERT(vec(2) == 3);

      auto vec32 = repmat(vec, vector2ui(1, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(3, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);
   }
};

TESTER_TEST_SUITE(TestArrayRepmat);
TESTER_TEST(testRepmat_simple);
TESTER_TEST(testRepmat_simple2);
TESTER_TEST(testRepmat_simple3);
TESTER_TEST(testRepmat_strided);
TESTER_TEST_SUITE_END();
