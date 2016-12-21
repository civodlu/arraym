#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestMatrixCov
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
      for (size_t n = 0; n < 50; ++n)
      {
         MatrixT a(500 + generateUniformDistribution<size_t>(2, 100), generateUniformDistribution<size_t>(2, 100));
         random(a);
         cov(a);
         //TESTER_ASSERT(error < 1e-3);
      }
   }

   void test_known()
   {
      // - TODO easily create a 1D matrix from a vector
      // - TODO repmat

      // http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
      Matrix<double> points(5, 3);
      points = 
      {
            4.0, 2.0, 0.6,
            4.2, 2.1, 0.59,
            3.9, 2.0, 0.58,
            4.3, 2.1, 0.62,
            4.1, 2.2, 0.63
      };

      std::cout << "M=" << points << std::endl;

      const auto covariance = cov(points);
      std::cout << "Cov=" << std::endl << covariance << std::endl;

      TESTER_ASSERT(covariance.shape() == vector2ui(3, 3));
      TESTER_ASSERT(equal<double>(covariance(0, 0), 0.025, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(0, 1), 0.0075, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(0, 2), 0.00175, 1e-5));

      TESTER_ASSERT(equal<double>(covariance(1, 0), 0.0075, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(1, 1), 0.0070, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(1, 2), 0.00135, 1e-5));

      TESTER_ASSERT(equal<double>(covariance(2, 0), 0.00175, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(2, 1), 0.00135, 1e-5));
      TESTER_ASSERT(equal<double>(covariance(2, 2), 0.00043, 1e-5));
   }
};

TESTER_TEST_SUITE(TestMatrixCov);
//TESTER_TEST(test_random);
//TESTER_TEST(test_known);
TESTER_TEST_SUITE_END();
