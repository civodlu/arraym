
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayExp
{
   void test_expr_add_array_array_impl()
   {
      test_expr_add_array_array_impl<Matrix<float>>();
      test_expr_add_array_array_impl<Matrix<int>>();
      test_expr_add_array_array_impl<Matrix_row_major<float>>();
   }

   template <class Array>
   void test_expr_add_array_array_impl()
   {
      using value_type = typename Array::value_type;
      Array a(2, 2);
      a = { 1, 2, 
            3, 4 };
      //a = transpose(a);

      auto result = a * 3.0 + int(1) + a / 1.0f - 3;
      TESTER_ASSERT(result.shape() == vector2ui(2, 2));
      TESTER_ASSERT(result(0, 0) == 1 * 4 + 1 - 3);
      TESTER_ASSERT(result(0, 1) == 2 * 4 + 1 - 3);
      TESTER_ASSERT(result(1, 0) == 3 * 4 + 1 - 3);
      TESTER_ASSERT(result(1, 1) == 4 * 4 + 1 - 3);
   }
};

TESTER_TEST_SUITE(TestArrayExp);
TESTER_TEST(test_expr_add_array_array_impl);
TESTER_TEST_SUITE_END();