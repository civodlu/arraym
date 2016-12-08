#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestMatrixSvd
{
   template <class Array>
   void random(Array& array)
   {
      auto f = [](typename Array::value_type) { return generateUniformDistribution<float>(-5, 5); };

      auto op = [&](typename Array::value_type* y_pointer, ui32 y_stride, ui32 nb_elements) {
         NAMESPACE_NLL::details::apply_naive1(y_pointer, y_stride, nb_elements, f);
      };

      iterate_array(array, op);
   }

   void test_random()
   {
      test_random_impl<Matrix_row_major<float>>();
      test_random_impl<Matrix_column_major<float>>();
      test_random_impl<Matrix_row_major<double>>();
      test_random_impl<Matrix_column_major<double>>();
   }

   template <class MatrixT>
   void test_random_impl()
   {
      for (size_t n = 0; n < 500; ++n)
      {
         MatrixT a(generateUniformDistribution<size_t>(2, 100), generateUniformDistribution<size_t>(2, 100));
         random(a);

         MatrixT u, vt;
         Vector<MatrixT::value_type> s;

         svd(a, u, s, vt);
         const auto a_reconstructed = u * svd_construct_s(u, s, vt) * vt;
         const auto diff            = a_reconstructed - a;
         const auto error           = norm2(diff);
         TESTER_ASSERT(error < 1e-3);

         for (size_t n = 0; n < s.size() - 1; ++n)
         {
            TESTER_ASSERT(s(n) >= s(n + 1));
         }
      }
   }
};

TESTER_TEST_SUITE(TestMatrixSvd);
TESTER_TEST(test_random);
TESTER_TEST_SUITE_END();
