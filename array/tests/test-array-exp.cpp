#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayExp
{
   void test_matrix_mul_matrix()
   {
      using Vectorf = Array<float, 1>;

      Vectorf a1(2);
      a1(0) = 2;
      a1(1) = 3;

      Vectorf a2(2);
      a2(0) = 12;
      a2(1) = 13;

      Vectorf a3 = a1 + a2;
      std::cout << a3(0) << std::endl;
   }
};

TESTER_TEST_SUITE(TestArrayExp);
TESTER_TEST(test_matrix_mul_matrix);
TESTER_TEST_SUITE_END();