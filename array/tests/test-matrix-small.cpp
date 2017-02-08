#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestMatrixSmall
{
   void test_allocation()
   {
      using matrix_type = MatrixSmall_row_major<float, 16>;
      
      matrix_type m(vector2ui(3, 3));
      matrix_type v(vector2ui(3, 1));

      m = { 1, 2, 3,
         4, 5, 6,
         7, 8, 9 };
      v = { 10, 11, 12 };

      static_assert(!array_use_naive<matrix_type>::value, "should use BLAS");
      static_assert(array_use_blas<matrix_type>::value, "should use BLAS");
      static_assert(is_matrix<matrix_type>::value, "should be a matrix!");
      const auto result = m * v;

      TESTER_ASSERT(result.shape() == vector2ui(3, 1));
      TESTER_ASSERT(result(0, 0) == 68);
      TESTER_ASSERT(result(1, 0) == 167);
      TESTER_ASSERT(result(2, 0) == 266);
   }
};

TESTER_TEST_SUITE(TestMatrixSmall);
TESTER_TEST(test_allocation);
TESTER_TEST_SUITE_END();