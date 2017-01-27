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
      std::cout << cpu_1 << std::endl;

      cpu_array_type cpu_2 = gpu_1;
      std::cout << cpu_2 << std::endl;
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
      std::cout << result_cpu << std::endl;
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
      std::cout << result_cpu2 << std::endl;
      TESTER_ASSERT(result_cpu2.shape() == vector2ui(2, 1));
      TESTER_ASSERT(result_cpu2(0, 0) == 30 + 5);
      TESTER_ASSERT(result_cpu2(1, 0) == 40 + 6);
   }

   void test_construction()
   {
      using gpu_array_type = Array_cuda_column_major<float, 2>;

      gpu_array_type gpu = gpu_array_type({ 2, 3 });
      gpu = { 10, 20, 30, 40, 50, 60 };

   }
};

TESTER_TEST_SUITE(TestArrayCuda);
TESTER_TEST(test_basic_array);
TESTER_TEST(test_memoryBase);
TESTER_TEST(test_op_add);
TESTER_TEST(test_construction);
TESTER_TEST_SUITE_END();

#endif