#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestStaticVector
{
   void test_round()
   {
      StaticVector<float, 6> i(-0.51f, -0.49f, -0.1f, 0.4f, 0.6f, 1.6f);
      auto r = round<int>(i);

      TESTER_ASSERT(r[0] == -1);
      TESTER_ASSERT(r[1] == 0);
      TESTER_ASSERT(r[2] == 0);
      TESTER_ASSERT(r[3] == 0);
      TESTER_ASSERT(r[4] == 1);
      TESTER_ASSERT(r[5] == 2);
   }

   void test_conversion()
   {
      StaticVector< ui8, 1> v_ui8(127);

      StaticVector< float, 1> v_f(v_ui8);
      TESTER_ASSERT(v_f[0] == 127);
   }
};

TESTER_TEST_SUITE(TestStaticVector);
TESTER_TEST(test_round);
TESTER_TEST(test_conversion);
TESTER_TEST_SUITE_END();
