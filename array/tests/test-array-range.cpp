#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using vector3ui = StaticVector<ui32, 3>;
using vector2ui = StaticVector<ui32, 2>;
using vector1ui = StaticVector<ui32, 1>;

struct TestArrayRange
{
   void test_range()
   {
      static_assert(Array<float, 2>::is_range_list<Range, Range>::value, "<Range, Range>");
      static_assert(Array<float, 2>::is_range_list<const Range, Range>::value, "<Range, Range>");
      static_assert(Array<float, 2>::is_range_list<const Range, Range&>::value, "<Range, Range>");

      // not the good number of parameters
      static_assert(!Array<float, 2>::is_range_list<const Range>::value, "<Range>");
      static_assert(!Array<float, 2>::is_range_list<const Range, Range, Range>::value, "<Range, Range, Range>");

      // not the good type
      static_assert(!Array<float, 2>::is_range_list<const Range, int>::value, "<Range, int>");
      static_assert(!Array<float, 2>::is_range_list<int, const Range>::value, "<int, Range>");
   }

   void test_array_range()
   {
      using Array = Array<int, 2>;

      Array a1(2, 3);
      a1 = { 1, 2, 3, 4, 5, 6 };

      auto a2 = a1(R(0, 1), R(0, 0));
      TESTER_ASSERT(a2.shape() == vector2ui(2, 1));
      TESTER_ASSERT(a2(0, 0) == a1(0, 0));
      TESTER_ASSERT(a2(1, 0) == a1(1, 0));
      std::cout << "a2=" << a2 << std::endl;

      auto a3 = a1(R(1, 1), R(0, 1));
      TESTER_ASSERT(a3.shape() == vector2ui(1, 2));
      TESTER_ASSERT(a3(0, 0) == a1(1, 0));
      TESTER_ASSERT(a3(0, 1) == a1(1, 1));
   }

   void test_array_range_negative()
   {
      using Array = Array<int, 2>;

      Array a1(2, 3);
      a1 = { 1, 2,
             3, 4,
             5, 6 };

      {
         auto a2 = a1(R(-2, -1), R(0, 0));
         TESTER_ASSERT(a2.shape() == vector2ui(2, 1));
         TESTER_ASSERT(a2(0, 0) == a1(0, 0));
         TESTER_ASSERT(a2(1, 0) == a1(1, 0));
      }

      {
         auto a2 = a1(R(1, 1), R(-2, -1));
         TESTER_ASSERT(a2.shape() == vector2ui(1, 2));
         TESTER_ASSERT(a2(0, 0) == a1(1, 1));
         TESTER_ASSERT(a2(0, 1) == a1(1, 2));
      }
   }

   void test_array_range_all()
   {
      using Array = Array<int, 2>;

      Array a1(2, 3);
      a1 = { 1, 2,
         3, 4,
         5, 6 };

      {
         auto a2 = a1(rangeAll, R(0, 0));
         TESTER_ASSERT(a2.shape() == vector2ui(2, 1));
         TESTER_ASSERT(a2(0, 0) == a1(0, 0));
         TESTER_ASSERT(a2(1, 0) == a1(1, 0));
      }

      {
         auto a2 = a1(R(1, 1), rangeAll);
         TESTER_ASSERT(a2.shape() == vector2ui(1, 3));
         TESTER_ASSERT(a2(0, 0) == a1(1, 0));
         TESTER_ASSERT(a2(0, 1) == a1(1, 1));
         TESTER_ASSERT(a2(0, 2) == a1(1, 2));
      }
   }

   void test_constarray_range()
   {
      using Array = Array<int, 2>;

      Array ax(2, 3);
      ax = { 1, 2, 3, 4, 5, 6 };
      const Array& a1 = ax;

      auto a2 = a1(R(0, 1), R(0, 0));
      TESTER_ASSERT(a2.shape() == vector2ui(2, 1));
      TESTER_ASSERT(a2(0, 0) == a1(0, 0));
      TESTER_ASSERT(a2(1, 0) == a1(1, 0));
      std::cout << "a2=" << a2 << std::endl;
      
      auto a3 = a1(R(1, 1), R(0, 1));
      TESTER_ASSERT(a3.shape() == vector2ui(1, 2));
      TESTER_ASSERT(a3(0, 0) == a1(1, 0));
      TESTER_ASSERT(a3(0, 1) == a1(1, 1));
   }

   void test_constsubarray()
   {
      using Array = Array<int, 2>;

      Array ax(2, 3);
      ax = { 1, 2, 3, 4, 5, 6 };
      const Array& a1 = ax;

      auto sa1 = a1(vector2ui(0, 0), vector2ui(1, 2));
      TESTER_ASSERT(sa1.shape() == vector2ui(1, 2));
      TESTER_ASSERT(sa1(0, 0) == a1(0, 1));
      TESTER_ASSERT(sa1(0, 1) == a1(0, 2));
   }

   void test_range_op()
   {
      {
         const auto r = Range(1, 3) & Range(2, 10);
         TESTER_ASSERT(r.min == 2);
         TESTER_ASSERT(r.max == 3);
      }

      {
         const auto r = Range(1, 3) | Range(2, 10);
         TESTER_ASSERT(r.min == 1);
         TESTER_ASSERT(r.max == 10);
      }

      {
         const auto r = Range(1, 3) + 2;
         TESTER_ASSERT(r.min == 3);
         TESTER_ASSERT(r.max == 5);
      }

      {
         const auto r = Range(1, 3) - 2;
         TESTER_ASSERT(r.min == -1);
         TESTER_ASSERT(r.max == 1);
      }
   }
};

TESTER_TEST_SUITE(TestArrayRange);
TESTER_TEST(test_range);
TESTER_TEST(test_array_range);
TESTER_TEST(test_array_range_negative);
TESTER_TEST(test_array_range_all);
TESTER_TEST(test_constarray_range);
TESTER_TEST(test_range_op);
TESTER_TEST_SUITE_END();