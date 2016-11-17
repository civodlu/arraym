#pragma warning(disable:4244)

#include <array/blas-dispatcher.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayExp
{
   void test_matrix_mul_matrix()
   {
      
   }
};

TESTER_TEST_SUITE(TestArrayExp);
TESTER_TEST(test_matrix_mul_matrix);
TESTER_TEST_SUITE_END();