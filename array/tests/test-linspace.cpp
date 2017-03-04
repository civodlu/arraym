#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayLinspace
{
   void test_1d_simple()
   {
      auto a1 = linspace(1.0f, 5.0f, 5);
      TESTER_ASSERT(a1.shape() == vector1ui(5));

      TESTER_ASSERT(a1(0) == 1);
      TESTER_ASSERT(a1(1) == 2);
      TESTER_ASSERT(a1(2) == 3);
      TESTER_ASSERT(a1(3) == 4);
      TESTER_ASSERT(a1(4) == 5);
      std::cout << a1 << std::endl;
   }

   void test_1d_neg()
   {
      auto a1 = linspace(0.0f, 3.0f, 7);
      TESTER_ASSERT(a1.shape() == vector1ui(7));

      std::cout << a1 << std::endl;

      TESTER_ASSERT(a1(0) == 0.0f);
      TESTER_ASSERT(a1(1) == 0.5f);
      TESTER_ASSERT(a1(2) == 1.0f);
      TESTER_ASSERT(a1(3) == 1.5f);
      TESTER_ASSERT(a1(4) == 2.0f);
      TESTER_ASSERT(a1(5) == 2.5f);
      TESTER_ASSERT(a1(6) == 3.0f);
   }

   void test_1d_neg2()
   {
      auto a1 = linspace(-1.0f, 2.0f, 7);
      TESTER_ASSERT(a1.shape() == vector1ui(7));

      std::cout << a1 << std::endl;

      TESTER_ASSERT(a1(0) == -1.0f);
      TESTER_ASSERT(a1(1) == -0.5f);
      TESTER_ASSERT(a1(2) == 0.0f);
      TESTER_ASSERT(a1(3) == 0.5f);
      TESTER_ASSERT(a1(4) == 1.0f);
      TESTER_ASSERT(a1(5) == 1.5f);
      TESTER_ASSERT(a1(6) == 2.0f);
   }

   void test_1d_int()
   {
      // the slope < 1, make sure we have a good approximation
      auto a1 = linspace<int>(1, 5, 9);
      TESTER_ASSERT(a1.shape() == vector1ui(9));
      std::cout << a1 << std::endl;

      TESTER_ASSERT(a1(0) == 1); // 1.0
      TESTER_ASSERT(a1(1) == 1); // 1.5
      TESTER_ASSERT(a1(2) == 2); // 2.0
      TESTER_ASSERT(a1(3) == 2); // 2.5
      TESTER_ASSERT(a1(4) == 3); // 3.0
      TESTER_ASSERT(a1(5) == 3); // 3.5
      TESTER_ASSERT(a1(6) == 4); // 4.0
      TESTER_ASSERT(a1(7) == 4); // 4.5
      TESTER_ASSERT(a1(8) == 5); // 5.0
   }
};

TESTER_TEST_SUITE(TestArrayLinspace);
TESTER_TEST(test_1d_simple);
TESTER_TEST(test_1d_neg);
TESTER_TEST(test_1d_neg2);
TESTER_TEST(test_1d_int);
TESTER_TEST_SUITE_END();
