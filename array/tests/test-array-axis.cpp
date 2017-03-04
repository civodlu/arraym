#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

struct TestArrayAxis
{
   struct dummy_op
   {
      template <class T, size_t N, class Config>
      double operator()(const Array<T, N, Config>&) const
      {
         return 0.0;
      }
   };

   void test_axis_function()
   {
#ifdef WITH_CUDA
      test_axis_function_impl<Array_cuda_column_major<float, 2>>();
#endif
      test_axis_function_impl<Array_row_major<float, 2>>();
      test_axis_function_impl<Array_row_major<int, 2>>();

      test_axis_function_impl<Array_column_major<float, 2>>();
      test_axis_function_impl<Array_column_major<int, 2>>();

      test_axis_function_impl<Array_row_major_multislice<float, 2>>();
      test_axis_function_impl<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_axis_function_impl()
   {
      {
         // test return type inference
         using R = function_return_type<typename Array::value_type, Array::RANK, typename Array::Config, dummy_op>;
         static_assert(std::is_same<R, double>::value, "must have the same type!");

         dummy_op dummy;
         auto result = constarray_axis_apply_function(Array(), 1, dummy);
         using R2    = typename decltype(result)::value_type;
         static_assert(std::is_same<double, R2>::value, "array type should be defined by summarizing function's return type");
      }

      Array a(2, 3);
      a = {1, 2, 3, 4, 5, 6};

      details::adaptor_mean f;
      auto result = constarray_axis_apply_function(a, 1, f);

      TESTER_ASSERT(result.shape() == vector1ui(2));
      TESTER_ASSERT(result(0) == (a(0, 0) + a(0, 1) + a(0, 2)) / 3);
      TESTER_ASSERT(result(1) == (a(1, 0) + a(1, 1) + a(1, 2)) / 3);

      auto result2 = mean(a, 1);
      TESTER_ASSERT(result2.shape() == result.shape());
      TESTER_ASSERT(result2(0) == result(0));
      TESTER_ASSERT(result2(1) == result(1));
   }

   void test_axis_function2()
   {
#ifdef WITH_CUDA
      test_axis_function_impl2<Array_cuda_column_major<float, 2>>();
#endif
      test_axis_function_impl2<Array_row_major<float, 2>>();
      test_axis_function_impl2<Array_row_major<int, 2>>();

      test_axis_function_impl2<Array_column_major<float, 2>>();
      test_axis_function_impl2<Array_column_major<int, 2>>();

      test_axis_function_impl2<Array_row_major_multislice<float, 2>>();
      test_axis_function_impl2<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_axis_function_impl2()
   {
      Array a(2, 3);
      a = {1, 2, 3, 4, 5, 6};

      details::adaptor_mean f;
      auto result = constarray_axis_apply_function(a, 0, f);

      TESTER_ASSERT(result.shape() == vector1ui(3));
      TESTER_ASSERT(result(0) == (a(0, 0) + a(1, 0)) / 2);
      TESTER_ASSERT(result(1) == (a(0, 1) + a(1, 1)) / 2);
      TESTER_ASSERT(result(2) == (a(0, 2) + a(1, 2)) / 2);

      auto result2 = mean(a, 0);
      TESTER_ASSERT(result2.shape() == result.shape());
      TESTER_ASSERT(result2(0) == result(0));
      TESTER_ASSERT(result2(1) == result(1));
      TESTER_ASSERT(result2(2) == result(2));

      {
         auto max_result = max(a, 0);
         TESTER_ASSERT(max_result.shape() == result.shape());
         TESTER_ASSERT(max_result(0) == 2);
         TESTER_ASSERT(max_result(1) == 4);
         TESTER_ASSERT(max_result(2) == 6);
      }

      {
         auto min_result = min(a, 0);
         TESTER_ASSERT(min_result.shape() == result.shape());
         TESTER_ASSERT(min_result(0) == 1);
         TESTER_ASSERT(min_result(1) == 3);
         TESTER_ASSERT(min_result(2) == 5);
      }

      {
         auto sum_result = sum(a, 0);
         TESTER_ASSERT(sum_result.shape() == result.shape());
         TESTER_ASSERT(sum_result(0) == 1 + 2);
         TESTER_ASSERT(sum_result(1) == 3 + 4);
         TESTER_ASSERT(sum_result(2) == 5 + 6);
      }
   }

   void test_rebind_dim()
   {
      test_rebind_dim_impl<Array<float, 3>>();
      test_rebind_dim_impl<Array<int, 2>>();
      test_rebind_dim_impl<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_rebind_dim_impl()
   {
      using Other = typename Array::template rebind_dim<Array::RANK - 1>::other;

      Other test;
      TESTER_ASSERT(test.shape() == typename Other::index_type());
   }
};

TESTER_TEST_SUITE(TestArrayAxis);
TESTER_TEST(test_axis_function);
TESTER_TEST(test_axis_function2);
TESTER_TEST(test_rebind_dim);
TESTER_TEST_SUITE_END();
