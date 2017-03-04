#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestStack
{
   template <class Array>
   void random(Array& array)
   {
      auto f = [](typename Array::value_type) { return generateUniformDistribution<float>(-5, 5); };

      auto op = [&](typename Array::pointer_type y_pointer, ui32 y_stride, ui32 nb_elements) {
         auto y_end = y_pointer + y_stride * nb_elements;
         for (; y_pointer != y_end; y_pointer += y_stride)
         {
            const typename Array::value_type value = f(*y_pointer);
            NAMESPACE_NLL::details::copy_naive(y_pointer, 1, &value, 1, 1);
         }
      };

      iterate_array(array, op);
   }

   void test_random()
   {
#ifdef WITH_CUDA
      test_random_impl<Array_cuda_column_major<float, 2>>();
#endif
      test_random_impl<Array<float, 6>>();
      test_random_impl<Array<double, 1>>();
      test_random_impl<Matrix_row_major<double>>();
   }

   template <class Array>
   void test_random_impl()
   {
      using index_type = typename Array::index_type;

      for (size_t n = 0; n < 100; ++n)
      {
         index_type shape;
         for (auto& value : shape)
         {
            value = generateUniformDistribution(1, 10);
         }

         Array array1(shape);
         Array array2(shape);
         Array array3(shape);
         random(array1);
         random(array2);
         random(array3);

         auto s = stack(array1, array2, array3);

         TESTER_ASSERT(s.shape()[Array::RANK] == 3);
         for (auto i : range(Array::RANK))
         {
            TESTER_ASSERT(s.shape()[i] == shape[i]);
         }
      }
   }

   void test_norm2()
   {
#ifdef WITH_CUDA
      test_norm2_impl<Array_cuda_column_major<float, 2>>();
#endif
      test_norm2_impl<Array<float, 2>>();
      test_norm2_impl<Array<int, 2>>();
   }

   template <class Array>
   void test_norm2_impl()
   {
      Array a(2, 3);
      a                     = {1, 2, 3, 4, 5, 6};
      const double expected = std::sqrt(1 * 1 + 2 * 2 + 3 * 3 + 4 * 4 + 5 * 5 + 6 * 6);
      const double found    = norm2(a);
      TESTER_ASSERT(fabs(expected - found) < 1e-4);
   }
};

TESTER_TEST_SUITE(TestStack);
TESTER_TEST(test_random);
TESTER_TEST(test_norm2);
TESTER_TEST_SUITE_END();
