#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector1ui = StaticVector<ui32, 1>;
using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

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
#ifdef WITH_CUDA
      test_random_impl<Matrix_column_major<float>>();
#endif
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
      test_known_impl<Matrix_column_major<double>>();
      test_known_impl<Matrix_column_major<float>>();
      test_known_impl<Matrix_row_major<double>>();
#ifdef WITH_CUDA
      test_known_impl<Matrix_column_major<float>>();
#endif
   }

   template <class matrix_type>
   void test_known_impl()
   {
      // http://www.itl.nist.gov/div898/handbook/pmc/section5/pmc541.htm
      matrix_type points(5, 3);
      points = 
      {
            4.0, 2.0, 0.6,
            4.2, 2.1, 0.59,
            3.9, 2.0, 0.58,
            4.3, 2.1, 0.62,
            4.1, 2.2, 0.63
      };


      using T = typename matrix_type::value_type;
      const size_t N = 2;
      using Config = typename matrix_type::Config;
      
      using T1 = typename  Array<T, N, Config>::template rebind_dim<N - 1>::other;

      const auto covariance = cov(points);
      
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

   void test_vector_asMatrix_rowMajor()
   {
      Vector<int> vec(5);
      vec = { 1, 2, 3, 4, 5 };

      auto m = as_matrix_row_major(vec, vector2ui(5, 1));
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same buffer!
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(3, 0) == 4);
      TESTER_ASSERT(m(4, 0) == 5);

      auto vec2 = as_vector(m);
      TESTER_ASSERT(vec2.shape() == vec.shape());
      TESTER_ASSERT(&vec2(0) == &vec(0));
   }

   void test_vector_asMatrix_colMajor()
   {
      Vector<int> vec(5);
      vec = { 1, 2, 3, 4, 5 };

      auto m = as_matrix_column_major(vec, vector2ui(5, 1));
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same buffer!
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(3, 0) == 4);
      TESTER_ASSERT(m(4, 0) == 5);

      auto vec2 = as_vector(m);
      TESTER_ASSERT(vec2.shape() == vec.shape());
      TESTER_ASSERT(&vec2(0) == &vec(0));
   }

   void test_vector_asArrayRowMajor()
   {
      Vector<int> vec(6);
      vec = { 1, 2, 3, 4, 5, 6 };

      auto m = as_array_row_major(vec, vector3ui(3, 2, 1));
      TESTER_ASSERT(m.shape() == vector3ui(3, 2, 1));
      TESTER_ASSERT(m(0, 0, 0) == 1);
      TESTER_ASSERT(m(1, 0, 0) == 2);
      TESTER_ASSERT(m(2, 0, 0) == 3);
      TESTER_ASSERT(m(0, 1, 0) == 4);
      TESTER_ASSERT(m(1, 1, 0) == 5);
      TESTER_ASSERT(m(2, 1, 0) == 6);
      TESTER_ASSERT(&m(0, 0, 0) == &vec(0)); // must share the same pointer
   }

   void test_vector_asArrayColumnMajor()
   {
      Vector<int> vec(6);
      vec = { 1, 2, 3, 4, 5, 6 };

      auto m = as_array_column_major(vec, vector2ui(3, 2));

      TESTER_ASSERT(m.shape() == vector2ui(3, 2));
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(0, 1) == 2);
      TESTER_ASSERT(m(1, 0) == 3);
      TESTER_ASSERT(m(1, 1) == 4);
      TESTER_ASSERT(m(2, 0) == 5);
      TESTER_ASSERT(m(2, 1) == 6);
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same pointer
   }

   void test_vector_asArrayColumnMajor_stride()
   {
      int values[] =
      {
         1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0
      };

      using vector_type = Vector<int>;
      vector_type vec( vector_type::Memory( vector1ui( 6 ), values, vector1ui( 2 ) ) );

      auto m = as_array_column_major( vec, vector2ui( 3, 2 ) );

      TESTER_ASSERT( m.shape() == vector2ui( 3, 2 ) );
      TESTER_ASSERT( m( 0, 0 ) == 1 );
      TESTER_ASSERT( m( 0, 1 ) == 2 );
      TESTER_ASSERT( m( 1, 0 ) == 3 );
      TESTER_ASSERT( m( 1, 1 ) == 4 );
      TESTER_ASSERT( m( 2, 0 ) == 5 );
      TESTER_ASSERT( m( 2, 1 ) == 6 );
      TESTER_ASSERT( &m( 0, 0 ) == &vec( 0 ) ); // must share the same pointer
   }

   void test_vector_asArrayRowMajor_stride()
   {
      int values[] =
      {
         1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0
      };

      using vector_type = Vector<int>;
      vector_type vec(vector_type::Memory(vector1ui(6), values, vector1ui(2)));

      auto m = as_array_row_major(vec, vector2ui(3, 2));

      TESTER_ASSERT(m.shape() == vector2ui(3, 2));
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(0, 1) == 4);
      TESTER_ASSERT(m(1, 1) == 5);
      TESTER_ASSERT(m(2, 1) == 6);
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same pointer
   }

   void test_vector_asArray()
   {
      Array<int, 2> vec(3, 2);
      vec = { 1, 2, 3, 4, 5, 6 };

      auto m = as_array(vec, vector3ui(3, 2, 1));

      TESTER_ASSERT(m.shape() == vector3ui(3, 2, 1));
      TESTER_ASSERT(m(0, 0, 0) == 1);
      TESTER_ASSERT(m(1, 0, 0) == 2);
      TESTER_ASSERT(m(2, 0, 0) == 3);
      TESTER_ASSERT(m(0, 1, 0) == 4);
      TESTER_ASSERT(m(1, 1, 0) == 5);
      TESTER_ASSERT(m(2, 1, 0) == 6);
      TESTER_ASSERT(&m(0, 0, 0) == &vec(0, 0)); // must share the same pointer
   }

   void test_vector_asMatrix_rowMajor_stride()
   {
      int values[] = 
      {
         1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0
      };

      using vector_type = Vector<int>;
      vector_type vec(vector_type::Memory(vector1ui(5), values, vector1ui(2)));

      auto m = as_matrix_row_major(vec, vector2ui(5, 1));
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same buffer!
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(3, 0) == 4);
      TESTER_ASSERT(m(4, 0) == 5);
   }

   void test_vector_asMatrix_colMajor_stride()
   {
      int values[] =
      {
         1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0
      };

      using vector_type = Vector<int>;
      vector_type vec(vector_type::Memory(vector1ui(5), values, vector1ui(2)));

      auto m = as_matrix_column_major(vec, vector2ui(5, 1));
      TESTER_ASSERT(&m(0, 0) == &vec(0)); // must share the same buffer!
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(3, 0) == 4);
      TESTER_ASSERT(m(4, 0) == 5);
   }
};

TESTER_TEST_SUITE(TestMatrixCov);
TESTER_TEST(test_vector_asMatrix_rowMajor);
TESTER_TEST(test_vector_asMatrix_colMajor);
TESTER_TEST(test_vector_asArrayRowMajor);
TESTER_TEST(test_vector_asArrayColumnMajor);
TESTER_TEST(test_vector_asArray);
TESTER_TEST(test_vector_asMatrix_rowMajor_stride);
TESTER_TEST(test_vector_asMatrix_colMajor_stride);
TESTER_TEST( test_vector_asArrayColumnMajor_stride );
TESTER_TEST(test_vector_asArrayRowMajor_stride);
//TESTER_TEST(test_random);
TESTER_TEST(test_known);
TESTER_TEST_SUITE_END();
