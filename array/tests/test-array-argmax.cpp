#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayArg
{
   void test_simple_argmax()
   {
      using array_type = Array<float, 2>;

      array_type array(3, 2);
      array            = {1, 2, 3, 6, 5, 4};
      const auto index = argmax(array);

      TESTER_ASSERT(array(index) == 6);

      {
         auto sub = array(R(0, 2), R(0, 0));
         TESTER_ASSERT(sub(argmax(sub)) == 3);
      }

      {
         auto sub = array(R(0, 2), R(1, 1));
         TESTER_ASSERT(sub(argmax(sub)) == 6);
      }

      {
         auto sub = array(R(0, 0), R(0, 1));
         TESTER_ASSERT(sub(argmax(sub)) == 6);
      }
   }

   void test_simple_argmin()
   {
      using array_type = Array<float, 2>;

      array_type array(3, 2);
      array            = {2, 6, 3, 5, 1, 4};
      const auto index = argmin(array);
      TESTER_ASSERT(array(index) == 1);

      {
         auto sub = array(R(0, 2), R(0, 0));
         TESTER_ASSERT(sub(argmin(sub)) == 2);
      }

      {
         auto sub = array(R(0, 2), R(1, 1));
         TESTER_ASSERT(sub(argmin(sub)) == 1);
      }

      {
         auto sub = array(R(0, 0), R(0, 1));
         TESTER_ASSERT(sub(argmin(sub)) == 2);
      }
   }

   void test_simple_argmin_axis()
   {
      test_simple_argmin_axis_impl<Array<float, 2>>();
      test_simple_argmin_axis_impl<Array_column_major<float, 2>>();
      test_simple_argmin_axis_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_simple_argmin_axis_impl()
   {
      array_type array(3, 2);
      array                = {2, 6, 3, 5, -1, 4};
      const auto array_min = argmin(array, 1);

      TESTER_ASSERT(array_min.shape() == vector1ui(3));
      TESTER_ASSERT(array(array_min(0)) == 2);
      TESTER_ASSERT(array(array_min(1)) == -1);
      TESTER_ASSERT(array(array_min(2)) == 3);

      auto sub    = array(rangeAll, R(1, 1));
      auto result = argmin(sub, 0) + vector2ui(0, 1);
      TESTER_ASSERT(array(result(0)) == -1);
   }

   void test_simple_argmax_axis()
   {
      test_simple_argmax_axis_impl<Array<float, 2>>();
      test_simple_argmax_axis_impl<Array_column_major<float, 2>>();
      test_simple_argmax_axis_impl<Array_column_major<int, 2>>();
   }

   template <class array_type>
   void test_simple_argmax_axis_impl()
   {
      array_type array(3, 2);
      array                = {2, 6, 3, 5, -1, 4};
      const auto array_max = argmax(array, 1);

      TESTER_ASSERT(array_max.shape() == vector1ui(3));
      TESTER_ASSERT(array(array_max(0)) == 5);
      TESTER_ASSERT(array(array_max(1)) == 6);
      TESTER_ASSERT(array(array_max(2)) == 4);

      auto sub    = array(rangeAll, R(1, 1));
      auto result = argmax(sub, 0) + vector2ui(0, 1);
      TESTER_ASSERT(array(result(0)) == 5);
   }

   void test_axis_return_type()
   {
      using array_type = Array<float, 2>;
      using return_type = axis_apply_fun_type<array_type::value_type, array_type::RANK, array_type::Config, details::adaptor_max>;

      static_assert( std::is_same<array_type::value_type, return_type::value_type>::value, "unexpected type" );
   }
};

TESTER_TEST_SUITE(TestArrayArg);
TESTER_TEST( test_axis_return_type );
TESTER_TEST(test_simple_argmax);
TESTER_TEST(test_simple_argmin);
TESTER_TEST(test_simple_argmax_axis);
TESTER_TEST(test_simple_argmin_axis);
TESTER_TEST_SUITE_END();
