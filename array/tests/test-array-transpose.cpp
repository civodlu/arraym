#include <core/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayTranspose
{
   void test_transpose_generic_2_rowmajor()
   {
      Array<int, 2> a(3, 2);
      a = { 1, 2, 3,
            4, 5, 6 };
      std::cout << a << std::endl;

      auto at = transpose(a, vector2ui(1, 0));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector2ui(2, 3));
      TESTER_ASSERT(at(0, 0) == 1);
      TESTER_ASSERT(at(0, 1) == 2);
      TESTER_ASSERT(at(0, 2) == 3);
      TESTER_ASSERT(at(1, 0) == 4);
      TESTER_ASSERT(at(1, 1) == 5);
      TESTER_ASSERT(at(1, 2) == 6);
   }

   void test_transpose_generic_2_colmajor()
   {
      Array_column_major<int, 2> a(3, 2);
      a = { 1, 2, 3,
         4, 5, 6 };
      std::cout << a << std::endl;

      auto at = transpose(a, vector2ui(1, 0));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector2ui(2, 3));
      TESTER_ASSERT(at(0, 0) == 1);
      TESTER_ASSERT(at(0, 1) == 2);
      TESTER_ASSERT(at(0, 2) == 3);
      TESTER_ASSERT(at(1, 0) == 4);
      TESTER_ASSERT(at(1, 1) == 5);
      TESTER_ASSERT(at(1, 2) == 6);
   }

   void test_transpose_generic_2_rowmajor_strided()
   {
      Array<int, 2> a(6, 5);
      auto a_sub = a(vector2ui(1, 2), vector2ui(3, 3));
      a_sub = { 1, 2, 3,
                4, 5, 6 };
      std::cout << a_sub << std::endl;

      auto at = transpose(a_sub, vector2ui(1, 0));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector2ui(2, 3));
      TESTER_ASSERT(at(0, 0) == 1);
      TESTER_ASSERT(at(0, 1) == 2);
      TESTER_ASSERT(at(0, 2) == 3);
      TESTER_ASSERT(at(1, 0) == 4);
      TESTER_ASSERT(at(1, 1) == 5);
      TESTER_ASSERT(at(1, 2) == 6);
   }

   void test_transpose_generic_3_rowmajor()
   {
      Array<int, 3> a(3, 2, 1);
      a = { 1, 2, 3,
         4, 5, 6 };
      std::cout << a << std::endl;

      auto at = transpose(a, vector3ui(1, 0, 2));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector3ui(2, 3, 1));
      TESTER_ASSERT(at(0, 0, 0) == 1);
      TESTER_ASSERT(at(0, 1, 0) == 2);
      TESTER_ASSERT(at(0, 2, 0) == 3);
      TESTER_ASSERT(at(1, 0, 0) == 4);
      TESTER_ASSERT(at(1, 1, 0) == 5);
      TESTER_ASSERT(at(1, 2, 0) == 6);
   }

   void test_transpose_generic_3_rowmajor_2()
   {
      Array<int, 3> a(3, 2, 2);
      a = { 1, 2, 3,
         4, 5, 6,
      
         7, 8, 9,
         10, 11, 12 };
      std::cout << a << std::endl;

      auto at = transpose(a, vector3ui(1, 0, 2));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector3ui(2, 3, 2));
      TESTER_ASSERT(at(0, 0, 0) == 1);
      TESTER_ASSERT(at(0, 1, 0) == 2);
      TESTER_ASSERT(at(0, 2, 0) == 3);
      TESTER_ASSERT(at(1, 0, 0) == 4);
      TESTER_ASSERT(at(1, 1, 0) == 5);
      TESTER_ASSERT(at(1, 2, 0) == 6);

      TESTER_ASSERT(at(0, 0, 1) == 7);
      TESTER_ASSERT(at(0, 1, 1) == 8);
      TESTER_ASSERT(at(0, 2, 1) == 9);
      TESTER_ASSERT(at(1, 0, 1) == 10);
      TESTER_ASSERT(at(1, 1, 1) == 11);
      TESTER_ASSERT(at(1, 2, 1) == 12);
   }

   void test_transpose_generic_3_rowmajor_2_strided()
   {
      Array<int, 3> a_big(3 + 5, 2 + 6, 2 + 7);

      auto a = a_big(vector3ui(1, 2, 3), vector3ui(1 + 3 - 1, 2 + 2 - 1, 3 + 2 - 1));
      a = { 1, 2, 3,
         4, 5, 6,

         7, 8, 9,
         10, 11, 12 };
      std::cout << a << std::endl;

      auto at = transpose(a, vector3ui(1, 0, 2));
      std::cout << at << std::endl;

      TESTER_ASSERT(at.shape() == vector3ui(2, 3, 2));
      TESTER_ASSERT(at(0, 0, 0) == 1);
      TESTER_ASSERT(at(0, 1, 0) == 2);
      TESTER_ASSERT(at(0, 2, 0) == 3);
      TESTER_ASSERT(at(1, 0, 0) == 4);
      TESTER_ASSERT(at(1, 1, 0) == 5);
      TESTER_ASSERT(at(1, 2, 0) == 6);

      TESTER_ASSERT(at(0, 0, 1) == 7);
      TESTER_ASSERT(at(0, 1, 1) == 8);
      TESTER_ASSERT(at(0, 2, 1) == 9);
      TESTER_ASSERT(at(1, 0, 1) == 10);
      TESTER_ASSERT(at(1, 1, 1) == 11);
      TESTER_ASSERT(at(1, 2, 1) == 12);
   }
};

TESTER_TEST_SUITE(TestArrayTranspose);
TESTER_TEST(test_transpose_generic_2_rowmajor);
TESTER_TEST(test_transpose_generic_2_colmajor);
TESTER_TEST(test_transpose_generic_2_rowmajor_strided);
TESTER_TEST(test_transpose_generic_3_rowmajor);
TESTER_TEST(test_transpose_generic_3_rowmajor_2);
TESTER_TEST(test_transpose_generic_3_rowmajor_2_strided);
TESTER_TEST_SUITE_END();
