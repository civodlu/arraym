#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using vector3ui = StaticVector<ui32, 3>;
using vector2ui = StaticVector<ui32, 2>;
using vector1ui = StaticVector<ui32, 1>;

struct TestArrayRangeA
{
   void test_range()
   {
      static_assert(Array<float, 2>::is_range_list<RangeA, RangeA>::value, "<RangeA, RangeA>");
      static_assert(Array<float, 2>::is_range_list<const RangeA, RangeA>::value, "<RangeA, RangeA>");
      static_assert(Array<float, 2>::is_range_list<const RangeA, RangeA&>::value, "<RangeA, RangeA>");

      // not the good number of parameters
      static_assert(!Array<float, 2>::is_range_list<const RangeA>::value, "<RangeA>");
      static_assert(!Array<float, 2>::is_range_list<const RangeA, RangeA, RangeA>::value, "<RangeA, RangeA, RangeA>");

      // not the good type
      static_assert(!Array<float, 2>::is_range_list<const RangeA, int>::value, "<RangeA, int>");
      static_assert(!Array<float, 2>::is_range_list<int, const RangeA>::value, "<int, RangeA>");
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
         const auto r = range(1, 3) & range(2, 10);
         TESTER_ASSERT(r.begin() == 2);
         TESTER_ASSERT(r.end() == 3);
      }

      {
         const auto r = range(1, 3) | range(2, 10);
         TESTER_ASSERT(r.begin() == 1);
         TESTER_ASSERT(r.end() == 10);
      }

      {
         const auto r = range(1, 3) + 2;
         TESTER_ASSERT(r.begin() == 3);
         TESTER_ASSERT(r.end() == 5);
      }

      {
         const auto r = range(1, 3) - 2;
         TESTER_ASSERT(r.begin() == -1);
         TESTER_ASSERT(r.end() == 1);
      }
   }

   void test_rangefor()
   {
      auto r = range(1, 3);
      TESTER_ASSERT(*r.begin() == 1);
      TESTER_ASSERT(*r.end() == 3);

      int expected_i = -2;
      for (auto i : range(-2, 10))
      {
         TESTER_ASSERT(expected_i == i);
         ++expected_i;
      }
   }

   void test_rangefor_step()
   {
      int expected_i = -2;
      for (auto i : range(-2, 10).step(2))
      {
         TESTER_ASSERT(expected_i == i);
         expected_i += 2;
      }
   }

   void test_rangefor_max()
   {
      int expected_i = 0;
      for (auto i : range(10).step(2))
      {
         TESTER_ASSERT(expected_i == i);
         expected_i += 2;
      }
   }

   void test_rangefor_indices_vector()
   {
      std::vector<int> values = { 10, 11, 12, 13, 14, 15 };

      int expected_i = 0;
      for (auto i : indices(values).step(2))
      {
         TESTER_ASSERT(expected_i == i);
         expected_i += 2;
      }
      TESTER_ASSERT(expected_i == 6);
   }

   void test_rangefor_indices_initializer()
   {
      int expected_i = 0;
      for (auto i : indices({ 10, 11, 12, 13, 14, 15 }).step(2))
      {
         TESTER_ASSERT(expected_i == i);
         expected_i += 2;
      }
      TESTER_ASSERT(expected_i == 6);
   }

   void test_rangefor_indices_staticarray()
   {
      static const int values[] = { 10, 11, 12, 13, 14, 15 };

      int expected_i = 0;
      for (auto i : indices(values).step(2))
      {
         TESTER_ASSERT(expected_i == i);
         expected_i += 2;
      }
      TESTER_ASSERT(expected_i == 6);
   }

   void test_range_letters()
   {
      char expected_i = 'a';
      for (auto i : range('a', 'd'))
      {
         TESTER_ASSERT(i == expected_i);
         ++expected_i;
      }
   }

   void test_range_dynamic()
   {
      auto r = range(0, 10);
      std::cout << typeid(r).name() << std::endl;

      for (auto i : r)
      {
         TESTER_ASSERT(r.begin() == 0);
         TESTER_ASSERT(r.end() == 10);
      }
   }
};

TESTER_TEST_SUITE(TestArrayRangeA);
TESTER_TEST(test_rangefor);
TESTER_TEST(test_rangefor_step);
TESTER_TEST(test_rangefor_max);
TESTER_TEST(test_rangefor_indices_vector);
TESTER_TEST(test_rangefor_indices_initializer);
TESTER_TEST(test_rangefor_indices_staticarray);
TESTER_TEST(test_range);
TESTER_TEST(test_array_range);
TESTER_TEST(test_array_range_negative);
TESTER_TEST(test_array_range_all);
TESTER_TEST(test_constarray_range);
TESTER_TEST(test_range_op);
TESTER_TEST(test_range_letters);
TESTER_TEST(test_range_dynamic);
TESTER_TEST_SUITE_END();