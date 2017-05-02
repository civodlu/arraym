#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

struct TestLeastSquare
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

   void test_simple()
   {
      for (unsigned int n = 0; n < 1000; ++n)
      {
         srand(n);
         const auto nb   = generateUniformDistribution<size_t>(10, 20);
         const auto dim  = generateUniformDistribution<size_t>(3, 10);
         const auto bdim = generateUniformDistribution<size_t>(1, 5);

         Matrix<float> a(vector2ui(nb, dim));
         Matrix<float> x(vector2ui(dim, bdim));

         random(a);
         random(x);

         const auto b = a * x;

         auto x_found    = least_square(a, b);
         const auto b_ax = b - a * x_found;

         //std::cout << b_ax << std::endl;
         const double residual = norm2(b_ax);
         TESTER_ASSERT(residual < 1e-4f);
      }
   }

   void test_identityMatrix()
   {
      using matrix_type = Matrix_row_major<int>;
      auto m = identityMatrix<matrix_type>(3);
      TESTER_ASSERT(m.rows() == m.columns());
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 0);
      TESTER_ASSERT(m(2, 0) == 0);

      TESTER_ASSERT(m(0, 1) == 0);
      TESTER_ASSERT(m(1, 1) == 1);
      TESTER_ASSERT(m(2, 1) == 0);

      TESTER_ASSERT(m(0, 2) == 0);
      TESTER_ASSERT(m(1, 2) == 0);
      TESTER_ASSERT(m(2, 2) == 1);
   }
};

TESTER_TEST_SUITE(TestLeastSquare);
TESTER_TEST(test_simple);
TESTER_TEST(test_identityMatrix);
TESTER_TEST_SUITE_END();
