#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

DECLARE_NAMESPACE_NLL

DECLARE_NAMESPACE_NLL_END

struct TestIndexing
{
   void test_stdvector()
   {
      using array_type = Array<float, 2>;

      array_type a(3, 2);

      a = {1, 2, 3, 4, 5, 6};

      std::vector<vector2ui> indexes = {{0, 0}, {2, 0}, {2, 1}};
      auto r                         = loopkup(a, indexes);
      TESTER_ASSERT(r.shape() == vector1ui(3));
      TESTER_ASSERT(r(0) == 1);
      TESTER_ASSERT(r(1) == 3);
      TESTER_ASSERT(r(2) == 6);
   }

   void test_arrayindex()
   {
      using array_type = Array<float, 2>;

      array_type a(3, 2);

      a = {1, 2, 3, 4, 5, 6};

      Vector<vector2ui> indexes(3);

      indexes = {{0, 0}, {2, 0}, {2, 1}};
      auto r  = loopkup(a, indexes);
      TESTER_ASSERT(r.shape() == vector1ui(3));
      TESTER_ASSERT(r(0) == 1);
      TESTER_ASSERT(r(1) == 3);
      TESTER_ASSERT(r(2) == 6);
   }
};

TESTER_TEST_SUITE(TestIndexing);
TESTER_TEST(test_stdvector);
TESTER_TEST(test_arrayindex);
TESTER_TEST_SUITE_END();
