#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

DECLARE_NAMESPACE_NLL

DECLARE_NAMESPACE_NLL_END

struct TestArrayIO
{
   void test_io()
   {
      using array_type = Array_row_major_multislice<float, 2>;

      array_type a(3, 2);
      a = { 1, 2, 3, 4, 5, 6 };

      std::stringstream ss;
      a.write(ss);

      array_type a_cpy;
      a_cpy.read(ss);
      TESTER_ASSERT(a == a_cpy);
   }

   void test_io_different_ordering()
   {
      using array_type1 = Array_row_major<float, 2>;
      using array_type2 = Array_column_major<float, 2>;

      array_type1 a(3, 2);
      a = { 1, 2, 3, 4, 5, 6 };

      std::stringstream ss;
      a.write(ss);

      array_type2 a_cpy;
      a_cpy.read(ss);
      
      // here the array order depends on the array ordering of the reader!
      TESTER_ASSERT(a_cpy(0, 0) == 1);
      TESTER_ASSERT(a_cpy(0, 1) == 2);

      TESTER_ASSERT(a_cpy(1, 0) == 3);
      TESTER_ASSERT(a_cpy(1, 1) == 4);

      TESTER_ASSERT(a_cpy(2, 0) == 5);
      TESTER_ASSERT(a_cpy(2, 1) == 6);
   }
};

TESTER_TEST_SUITE(TestArrayIO);
TESTER_TEST(test_io);
TESTER_TEST(test_io_different_ordering);
TESTER_TEST_SUITE_END();