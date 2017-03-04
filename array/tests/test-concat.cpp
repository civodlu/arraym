#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayConcat
{
   void test_concat_vertical()
   {
      using array_type = Array<float, 2>;

      array_type a1(3, 2);
      a1 = {1, 2, 3, 4, 5, 6};
      array_type a2(3, 1);
      a2 = {7, 8, 9};

      auto r = concat(a1, a2, 1);

      TESTER_ASSERT(r.shape() == vector2ui(3, 3));

      array_type expected(3, 3);
      expected = {1, 2, 3, 4, 5, 6, 7, 8, 9};

      TESTER_ASSERT(norm2(r - expected) == 0);
   }

   void test_concat_horizontal()
   {
      using array_type = Array<float, 2>;

      array_type a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};
      array_type a2(1, 3);
      a2 = {7, 8, 9};

      auto r = concat(a2, a1, 0);

      TESTER_ASSERT(r.shape() == vector2ui(3, 3));

      array_type expected(3, 3);
      expected = {7, 1, 2, 8, 3, 4, 9, 5, 6};

      TESTER_ASSERT(norm2(r - expected) == 0);
   }
};

TESTER_TEST_SUITE(TestArrayConcat);
TESTER_TEST(test_concat_vertical);
TESTER_TEST(test_concat_horizontal);
TESTER_TEST_SUITE_END();
