#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestMatrixDet
{
   void test_known()
   {
      using matrix_type = Matrix_row_major<float>;

      {
         matrix_type a(3, 3);
         a = {
            8, 2, 3,
            4, 5, 6,
            7, 8, 9 };

         auto d = det(a);
         TESTER_ASSERT(fabs(d -(-21)) < 1e-5);
      }

      {
         matrix_type a(3, 3);
         a = {
            0, 2, 3,
            4, 5, 6,
            7, 8, 9 };

         auto d = det(a);
         TESTER_ASSERT(fabs(d - (3)) < 1e-5);
      }

      {
         matrix_type a(4, 4);
         a = {
            10, 20, 30, 40,
            25, 6, 7, 8,
            9, 20, 11, 12,
            1, 0, 0, 10 };

         auto d = det(a);
         TESTER_ASSERT(fabs(d - (84400)) < 1e-2);
      }

      {
         matrix_type a(4, 4);
         a = {
            10, 20, 30, 40,
            25, 6, 7, 8,
            9, -20, 11, 12,
            1, 0, 0, 10 };

         auto d = det(a);
         TESTER_ASSERT(fabs(d - (-1.8920e5)) < 1.0);
      }
   }

   void test_trace()
   {
      using matrix_type = Matrix_row_major<float>;

      matrix_type a(3, 3);
      a = {
         8, 2, 3,
         4, 5, 6,
         7, 8, 9 };

      auto d = trace(a);
      TESTER_ASSERT(fabs(d - (8+5+9)) < 1e-5);
   }
};

TESTER_TEST_SUITE(TestMatrixDet);
TESTER_TEST(test_known);
TESTER_TEST(test_trace);
TESTER_TEST_SUITE_END();
