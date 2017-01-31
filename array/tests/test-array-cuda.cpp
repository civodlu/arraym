#define ___TEST_ONLY___CUDA_ENABLE_SLOW_DEREFERENCEMENT
#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

#ifdef WITH_CUDA

using namespace NAMESPACE_NLL;

using vector1ui = StaticVector<ui32, 1>;
using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

#include <cuda.h>
#include <cuda_runtime.h>

//
// Specific tests for CUDA as the CPU based tests are using deferencement which is not allowed for GPU
//
struct TestArrayCuda
{
   void test_basic_array()
   {
      using gpu_array_type = Array_cuda_column_major<float, 2>;
      using cpu_array_type = Array_column_major<float, 2>;

      cpu_array_type cpu_1 = cpu_array_type({ 2, 3 });
      cpu_1 = { 1, 2, 3, 4, 5, 6 };

      gpu_array_type gpu_1 = cpu_1;

      cpu_array_type cpu_2 = gpu_1;
      TESTER_ASSERT(norm2(cpu_1 - cpu_2) == 0);
   }

   void test_memoryBase()
   {
      using cpu_array_type = Array_column_major<float, 2>;

      cpu_array_type cpu_1 = cpu_array_type({ 2, 3 });
      
      const auto ptr = array_base_memory(cpu_1);
      TESTER_ASSERT(ptr == &cpu_1({ 0, 0 }));
   }

   void test_op_add()
   {
      using gpu_array_type = Array_cuda_column_major<float, 2>;
      using cpu_array_type = Array_column_major<float, 2>;

      cpu_array_type cpu_1 = cpu_array_type({ 2, 3 });
      cpu_1 = { 1, 2, 3, 4, 5, 6 };

      cpu_array_type cpu_2 = cpu_array_type({ 2, 3 });
      cpu_2 = { 10, 20, 30, 40, 50, 60 };

      gpu_array_type gpu_1 = cpu_1;
      gpu_array_type gpu_2 = cpu_2;

      auto result_gpu = gpu_1 + gpu_2;
      cpu_array_type result_cpu = result_gpu;
      TESTER_ASSERT(result_cpu(0, 0) == 11);
      TESTER_ASSERT(result_cpu(1, 0) == 22);

      TESTER_ASSERT(result_cpu(0, 1) == 33);
      TESTER_ASSERT(result_cpu(1, 1) == 44);

      TESTER_ASSERT(result_cpu(0, 2) == 55);
      TESTER_ASSERT(result_cpu(1, 2) == 66);

      gpu_array_type gpu_1b = gpu_1(R(0, 2), R(2, 3));
      gpu_array_type gpu_2b = gpu_2(R(0, 2), R(1, 2));
      auto result_gpu2 = gpu_1b + gpu_2b;
      cpu_array_type result_cpu2 = result_gpu2;
      TESTER_ASSERT(result_cpu2.shape() == vector2ui(2, 1));
      TESTER_ASSERT(result_cpu2(0, 0) == 30 + 5);
      TESTER_ASSERT(result_cpu2(1, 0) == 40 + 6);
   }

   void test_construction()
   {
      using gpu_array_type = Array_cuda_column_major<float, 2>;
      using cpu_array_type = Array_column_major<float, 2>;

      gpu_array_type gpu = gpu_array_type({ 2, 3 });
      gpu = { 10, 20, 30, 40, 50, 60 };
      
      cpu_array_type cpu = gpu;
      TESTER_ASSERT(cpu.shape() == vector2ui(2, 3));
      TESTER_ASSERT(cpu(0, 0) == 10);
      TESTER_ASSERT(cpu(1, 0) == 20);

      TESTER_ASSERT(cpu(0, 1) == 30);
      TESTER_ASSERT(cpu(1, 1) == 40);

      TESTER_ASSERT(cpu(0, 2) == 50);
      TESTER_ASSERT(cpu(1, 2) == 60);
   }

   void test_slow_dereferencement()
   {
      using gpu_array_type = Array_cuda_column_major<float, 2>;
      using cpu_array_type = Array_column_major<float, 2>;

      gpu_array_type gpu = gpu_array_type({ 2, 3 });
      gpu = { 10, 20, 30, 40, 50, 60 };

      TESTER_ASSERT(gpu(0, 0) == 10);
      TESTER_ASSERT(gpu(1, 0) == 20);

      TESTER_ASSERT(gpu(0, 1) == 30);
      TESTER_ASSERT(gpu(1, 1) == 40);

      TESTER_ASSERT(gpu(0, 2) == 50);
      TESTER_ASSERT(gpu(1, 2) == 60);
   }

   void test_matrix_access()
   {
      using matrix_type_gpu = Matrix_cuda_column_major<float>;
      using matrix_type_cpu = Matrix_column_major<float>;

      auto gpu = matrix_type_gpu({ 2, 3 });
      gpu = { 10, 20, 30, 40, 50, 60 };

      TESTER_ASSERT(gpu.rows() == 2);
      TESTER_ASSERT(gpu.columns() == 3);

      // test indexing
      TESTER_ASSERT(gpu(0, 0) == 10);
      TESTER_ASSERT(gpu(0, 1) == 20);
      TESTER_ASSERT(gpu(0, 2) == 30);

      TESTER_ASSERT(gpu(1, 0) == 40);
      TESTER_ASSERT(gpu(1, 1) == 50);
      TESTER_ASSERT(gpu(1, 2) == 60);

      matrix_type_cpu cpu = gpu;

      // test order is as expected
      TESTER_ASSERT(cpu(0, 0) == 10);
      TESTER_ASSERT(cpu(0, 1) == 20);
      TESTER_ASSERT(cpu(0, 2) == 30);

      TESTER_ASSERT(cpu(1, 0) == 40);
      TESTER_ASSERT(cpu(1, 1) == 50);
      TESTER_ASSERT(cpu(1, 2) == 60);

      // test memory order is column major
      auto ptr = array_base_memory(cpu);
      TESTER_ASSERT(*ptr++ == 10);
      TESTER_ASSERT(*ptr++ == 40);
      TESTER_ASSERT(*ptr++ == 20);
      TESTER_ASSERT(*ptr++ == 50);
      TESTER_ASSERT(*ptr++ == 30);
      TESTER_ASSERT(*ptr++ == 60);
   }

   void testStridedTransform_unit()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;
      
      auto gpu_1 = Array_gpu(10);
      auto gpu_2 = Array_gpu(10);
      gpu_1 = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };

      auto ptr_1 = array_base_memory(gpu_1);
      auto ptr_2 = array_base_memory(gpu_2);
      NAMESPACE_NLL::details::cos(ptr_2, 1, ptr_1, 1, 10);

      for ( size_t n = 0; n < 10; ++n )
      {
          TESTER_ASSERT( fabs( gpu_2( n ) - std::cos( gpu_1( n ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_cos()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::cos( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - std::cos( gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_sin()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::sin( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - std::sin( gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_sqrt()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::sqrt( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - std::sqrt( gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_log()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::log( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - std::log( gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_exp()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 24 );
      auto gpu_2 = Array_gpu( 300 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 0, 0, 0, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::exp( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         const auto expected = std::exp( gpu_1( n * 2 ) );
         const auto found = gpu_2( n * 3 );
         TESTER_ASSERT( fabs( expected - found ) < 1e-2f );
      }
   }

   void testStridedTransform_nonunit_sqr()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::sqr( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - ( gpu_1( n * 2 ) * gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }

   void testStridedTransform_nonunit_abs()
   {
      using Array_gpu = Array_cuda_column_major<float, 1>;

      auto gpu_1 = Array_gpu( 20 );
      auto gpu_2 = Array_gpu( 30 );
      gpu_1 = { 1, 0, 2, 0, 3, 0, -4, 0, -5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0 };

      auto ptr_1 = array_base_memory( gpu_1 );
      auto ptr_2 = array_base_memory( gpu_2 );
      NAMESPACE_NLL::details::abs( ptr_2, 3, ptr_1, 2, 10 );

      for ( size_t n = 0; n < 10; ++n )
      {
         TESTER_ASSERT( fabs( gpu_2( n * 3 ) - std::abs( gpu_1( n * 2 ) ) ) < 1e-4f );
      }
   }
};

TESTER_TEST_SUITE(TestArrayCuda);
TESTER_TEST(test_basic_array);
TESTER_TEST(test_memoryBase);
TESTER_TEST(test_op_add);
TESTER_TEST(test_construction);
TESTER_TEST(test_slow_dereferencement);
TESTER_TEST(test_matrix_access);
TESTER_TEST(testStridedTransform_unit);
TESTER_TEST(testStridedTransform_nonunit_cos);
TESTER_TEST( testStridedTransform_nonunit_sin );
TESTER_TEST( testStridedTransform_nonunit_sqr );
TESTER_TEST( testStridedTransform_nonunit_sqrt );
TESTER_TEST( testStridedTransform_nonunit_log );
TESTER_TEST( testStridedTransform_nonunit_exp );
TESTER_TEST( testStridedTransform_nonunit_abs );
TESTER_TEST_SUITE_END();

#endif