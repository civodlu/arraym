
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayFill
{
   void test_fill_value_nostride()
   {
      using array_type = Array<float, 2>;
      array_type array(3, 2);

      array = {1, 2, 3, 4, 5, 6};
      fill_value(array, [](array_type::value_type value) { return 2 * value; });

      TESTER_ASSERT(array(0, 0) == 2);
      TESTER_ASSERT(array(1, 0) == 4);
      TESTER_ASSERT(array(2, 0) == 6);

      TESTER_ASSERT(array(0, 1) == 8);
      TESTER_ASSERT(array(1, 1) == 10);
      TESTER_ASSERT(array(2, 1) == 12);
   }

   void test_fill_value_strided()
   {
      using array_type = Array<float, 1>;
      array_type array(8);

      array = {1, 2, 3, 4, 5, 6, 7, 8};

      auto ref = array.subarray(vector1ui(0), vector1ui(7), vector1ui(2));
      fill_value(ref, [](array_type::value_type value) { return 2 * value; });

      TESTER_ASSERT(array(0) == 2);
      TESTER_ASSERT(array(1) == 2);
      TESTER_ASSERT(array(2) == 6);
      TESTER_ASSERT(array(3) == 4);
      TESTER_ASSERT(array(4) == 10);
      TESTER_ASSERT(array(5) == 6);
      TESTER_ASSERT(array(6) == 14);
      TESTER_ASSERT(array(7) == 8);
   }
};

TESTER_TEST_SUITE(TestArrayFill);
TESTER_TEST(test_fill_value_nostride);
TESTER_TEST(test_fill_value_strided);
TESTER_TEST_SUITE_END();
