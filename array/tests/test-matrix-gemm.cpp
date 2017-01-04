#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestMatrixGemm
{
   void test_gemm_columnMajor()
   {
      test_gemm_columnMajorImpl<Matrix_column_major<float>>();
      test_gemm_columnMajorImpl<Matrix_column_major<double>>();
   }

   template <class matrix_type>
   void test_gemm_columnMajorImpl()
   {
      // http://www.ibm.com/support/knowledgecenter/SSFHY8_5.5.0/com.ibm.cluster.essl.v5r5.essl100.doc/am5gr_hsgemm.htm
      using value_type = typename matrix_type::value_type;
      using memory_type = typename matrix_type::Memory;

      value_type A_values[] =
      {
         1.0,  2.0, -1.0, -1.0,  4.0,
         2.0,  0.0,  1.0,  1.0, -1.0,
         1.0, -1.0, -1.0,  1.0,  2.0,
        -3.0,  2.0,  2.0,  2.0,  0.0,
         4.0,  0.0, -2.0,  1.0, -1.0,
        -1.0, -1.0,  1.0, -3.0,  2.0,
         999, 999, 999, 999, 999,
         999, 999, 999, 999, 999
      };

      matrix_type A( memory_type( vector2ui(6, 5), A_values, vector2ui(5, 1) ) );

      std::cout << A << std::endl;

      value_type B_values[] =
      {
         1.0, -1.0,  0.0,  2.0,
         2.0,  2.0, -1.0, -2.0,
         1.0,  0.0, -1.0,  1.0,
        -3.0, -1.0,  1.0, -1.0,
         4.0,  2.0, -1.0,  1.0,
         999, 999, 999, 999
      };

      matrix_type B( memory_type( vector2ui( 5, 4 ), B_values, vector2ui( 4, 1 ) ) );

      std::cout << B << std::endl;

      value_type C_values[] =
      {
         0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5,
         0.5, 0.5, 0.5, 0.5,
         999, 999, 999, 999
      };

      matrix_type C( memory_type( vector2ui( 6, 4 ), C_values, vector2ui( 4, 1 ) ) );

      std::cout << C << std::endl;

      NAMESPACE_NLL::details::gemm<value_type>( 1, A, B, 2, C );
      std::cout << C << std::endl;


      matrix_type C_expected( 6, 4 );
      C_expected = {
         24.0, 13.0, -5.0, 3.0,
         -3.0, -4.0, 2.0, 4.0,
         4.0, 1.0, 2.0, 5.0,
         -2.0, 6.0, -1.0, -9.0,
         -4.0, -6.0, 5.0, 5.0,
         16.0, 7.0, -4.0, 7.0,
      };
      std::cout << C_expected << std::endl;

      const auto diff = C - C_expected;
      std::cout << diff << std::endl;

      const auto error = norm2( C - C_expected );

      TESTER_ASSERT( error < 1e-4 );
   }

   void test_gemm_inv()
   {
      test_gemm_invImpl<Matrix_column_major<float>>();
   }

   template <class Matrix>
   void test_gemm_invImpl()
   {

   }
};

TESTER_TEST_SUITE( TestMatrixGemm );
TESTER_TEST( test_gemm_inv );
TESTER_TEST( test_gemm_columnMajor );
TESTER_TEST_SUITE_END();
