#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

template <class T>
T f2x(T value)
{
   return value * 2;
}

struct TestArrayOpApply
{
   void test_array_apply_function()
   {
      test_array_apply_function_impl<Array<float, 2>>();
      test_array_apply_function_impl<Array_column_major<float, 2>>();
      test_array_apply_function_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_function_impl<Array<int, 2>>();
      test_array_apply_function_impl<Array_column_major<int, 2>>();
      test_array_apply_function_impl<Array_row_major_multislice<int, 2>>();
   }

   template <class Array>
   void test_array_apply_function_impl()
   {
      Array a1(2, 3);
      a1 = {1, 2, 3, 4, 5, 6};

      using other_type = typename Array::value_type;
      
      auto mul2 = [](typename Array::value_type value)
      {
         return static_cast<other_type>(value * 2);
      };
      
      using return_type = function_return_type_applied<typename Array::value_type, Array::RANK, typename Array::Config, decltype(mul2)>;
      using expected_type = typename Array::template rebind<other_type>::other;
      static_assert(std::is_same<return_type, expected_type>::value, "must be the same type!");

      const auto a1_fun = constarray_apply_function(a1, f2x<other_type>);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = static_cast<other_type>(a1(x, y) * 2);
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   void test_array_apply_functions()
   {
      test_array_apply_functions_cos_impl<Array<float, 2>>();
      test_array_apply_functions_cos_impl<Array_row_major_multislice<float, 2>>();

#ifdef WITH_CUDA
      test_array_apply_functions_cos_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_sin_impl<Array<float, 2>>();
      test_array_apply_functions_sin_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_sin_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_sqrt_impl<Array<float, 2>>();
      test_array_apply_functions_sqrt_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_sqrt_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_abs_impl<Array<float, 2>>();
      test_array_apply_functions_abs_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_abs_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_min_impl<Array<float, 2>>();
      test_array_apply_functions_min_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_min_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_max_impl<Array<float, 2>>();
      test_array_apply_functions_max_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_max_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_log_impl<Array<float, 2>>();
      test_array_apply_functions_log_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_log_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_exp_impl<Array<float, 2>>();
      test_array_apply_functions_exp_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_exp_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_mean_impl<Array<float, 2>>();
      test_array_apply_functions_mean_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_mean_impl<Array_cuda_column_major<float, 2>>();
#endif

      test_array_apply_functions_sqr_impl<Array<float, 2>>();
      test_array_apply_functions_sqr_impl<Array_row_major_multislice<float, 2>>();
#ifdef WITH_CUDA
      test_array_apply_functions_sqr_impl<Array_cuda_column_major<float, 2>>();
#endif

      // TODO Cuda implementation
      test_array_apply_functions_round_impl<Array<float, 2>>();
      test_array_apply_functions_round_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_saturate_impl<Array<float, 2>>();
      test_array_apply_functions_saturate_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_cast_impl<Array<float, 2>>();
      test_array_apply_functions_cast_impl<Array_row_major_multislice<float, 2>>();

      test_array_apply_functions_minmax_impl<Array<float, 2>>();
      test_array_apply_functions_minmax_impl<Array_row_major_multislice<float, 2>>();
   }

   template <class Array>
   void test_array_apply_functions_cos_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = cos(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::cos(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sin_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = sin(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::sin(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sqrt_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = sqrt(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::sqrt(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_abs_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = abs(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::abs(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_sqr_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = sqr(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = a1(x, y) * a1(x, y);
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_round_impl()
   {
      Array a1(2, 3);
      a1 = { 1.1, 2.6, 3.9, -0.1, -0.6, -0.9 };
      const auto a1_fun = round<int>(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::round(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_saturate_impl()
   {
      Array a1(2, 3);
      a1 = { 1.1, 2.6, 3.9, -0.1, -0.6, -0.9 };

      NAMESPACE_NLL::Array<double, 2> expected_a1(2, 3);
      expected_a1 = { 1.1, 2.6, 3.5, -0.1, -0.2, -0.2 };

      const auto a1_fun = saturate<double, float>(a1, -0.2f, 3.5f);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = expected_a1(x, y);
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_cast_impl()
   {
      Array a1(2, 3);
      a1 = { 1.1, 2.6, 3.9, -0.1, -0.6, -0.9 };
      const auto a1_expected_fun = [](typename Array::value_type value)
      {
         return static_cast<int>(value);
      };

      const auto a1_fun = cast<int>(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = a1_expected_fun(a1(x, y));
            const auto found = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_log_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = log(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::log(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_mean_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = mean(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = (1 + 2 + 3 + 4 + 5 + 6) / (typename Array::value_type)(6);
            const auto found    = a1_fun;
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_exp_impl()
   {
      Array a1(2, 3);
      a1                = {1, 2, 3, 4, 5, 6};
      const auto a1_fun = exp(a1);
      for (size_t y = 0; y < a1.shape()[1]; ++y)
      {
         for (size_t x = 0; x < a1.shape()[0]; ++x)
         {
            const auto expected = std::exp(a1(x, y));
            const auto found    = a1_fun(x, y);
            TESTER_ASSERT(std::abs(expected - found) < 1e-4f);
         }
      }
   }

   template <class Array>
   void test_array_apply_functions_min_impl()
   {
      Array a1(2, 3);
      a1                                      = {2, 1, 3, 4, 5, 6};
      const typename Array::value_type a1_fun = min(a1);
      TESTER_ASSERT(a1_fun == 1);
   }

   template <class Array>
   void test_array_apply_functions_minmax_impl()
   {
      Array a1(2, 3);
      a1 = { 2, 1, 3, 4, 5, 6 };
      const auto a1_fun = minmax(a1);
      TESTER_ASSERT(a1_fun.first == 1);
      TESTER_ASSERT(a1_fun.second == 6);
   }

   template <class Array>
   void test_array_apply_functions_max_impl()
   {
      Array a1(2, 3);
      a1                                      = {2, 1, 3, 4, 5, 6};
      const typename Array::value_type a1_fun = max(a1);
      TESTER_ASSERT(a1_fun == 6);
   }

   void test_array_argmax()
   {
      test_array_argmax_impl<Array<float, 2>>();
   }

   template <class T, size_t N, class Config>
   size_t argmax(const Array<T, N, Config>& array)
   {
      size_t index     = 0;
      T max_value      = std::numeric_limits<T>::lowest();
      size_t max_index = 0;
      auto f           = [&](T value) {
         if (value > max_value)
         {
            max_value = value;
            max_index = index;
         }

         ++index;
      };
      constarray_apply_function_inplace(array, f);
      return max_index;
   }

   template <class Array>
   void test_array_argmax_impl()
   {
      Array array(3, 2);
      array = {4, 2, 1, 5, 6, 3};

      auto index = argmax(array);
      TESTER_ASSERT(index == 4);
   }

   template <class T>
   struct Functor
   {
      T operator()(float) const
      {
         return T(0);
      }
   };

   void test_function_return_type()
   {
      auto f_int = [](float)->int
      {
         return int(0);
      };

      static_assert(std::is_same<int, function_return_type<float, decltype(f_int)>>::value, "wrong type!");
      static_assert(std::is_same<int, function_return_type<float, Functor<int> >>::value, "wrong type!");
   }

   void test_sqr_vec()
   {
      using vector3f = StaticVector<int, 3>;
      using array_type = Array<vector3f, 1>;

      array_type array(3);
      array =
      {
         vector3f(1, 2, 3), vector3f(4, 5, 6), vector3f(7, 8, 9)
      };

      const auto value_sqr = sqr(array);
      TESTER_ASSERT(value_sqr.shape()[0] == 3);
      TESTER_ASSERT(value_sqr[0] == vector3f(1 * 1, 2 * 2, 3 * 3));
      TESTER_ASSERT(value_sqr[1] == vector3f(4 * 4, 5 * 5, 6 * 6));
      TESTER_ASSERT(value_sqr[2] == vector3f(7 * 7, 8 * 8, 9 * 9));
   }

   void test_norm2_vec()
   {
      using vector3f = StaticVector<int, 3>;
      using array_type = Array<vector3f, 1>;

      array_type array(3);
      array =
      {
         vector3f(1, 2, 3), vector3f(4, 5, 6), vector3f(7, 8, 9)
      };

      const double value = norm2(array);
      const double expected = std::sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6 + 7 * 7 + 8 * 8 + 9 * 9);
      TESTER_ASSERT(equal(expected, value, 1e-4));
   }

   void test_norm2sqr()
   {
      using array_type = Array<float, 1>;

      array_type array(3);
      array =
      {
         1, 2, 3
      };

      const float value = norm2sqr(array);
      const float expected = 1 * 1 + 2 * 2 + 3 * 3;
      TESTER_ASSERT(equal(expected, value));
   }

   void test_norm2_elementwise()
   {
      using vector3f = StaticVector<int, 3>;
      using array_type = Array<vector3f, 1>;

      array_type array(3);
      array =
      {
         vector3f(1, 2, 3), vector3f(4, 5, 6), vector3f(7, 8, 9)
      };

      std::cout << norm2_elementwise(array) << std::endl;
      const auto value = norm2_elementwise(array);
      TESTER_ASSERT(value.shape() == vector1ui(3));
      TESTER_ASSERT(equal<float>(value[0], std::sqrt(1.0f * 1 + 2 * 2 + 3 * 3), 1e-4f));
      TESTER_ASSERT(equal<float>(value[1], std::sqrt(4.0f * 4 + 5 * 5 + 6 * 6), 1e-4f));
      TESTER_ASSERT(equal<float>(value[2], std::sqrt(7.0f * 7 + 8 * 8 + 9 * 9), 1e-4f));
   }
};

TESTER_TEST_SUITE(TestArrayOpApply);
TESTER_TEST(test_array_apply_function);
TESTER_TEST(test_array_apply_functions);
TESTER_TEST(test_array_argmax);
TESTER_TEST(test_function_return_type);
TESTER_TEST(test_sqr_vec);
TESTER_TEST(test_norm2_vec);
TESTER_TEST(test_norm2sqr);
TESTER_TEST(test_norm2_elementwise);
TESTER_TEST_SUITE_END();
