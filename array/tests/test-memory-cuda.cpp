#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector1ui = StaticVector<ui32, 1>;
using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

#include <cuda.h>
#include <cuda_runtime.h>

struct TestMemoryCuda
{
   void testInit_cte()
   {
      using memory_type = Memory_gpu_cuda<float, 1>;
      using memory_type_cpu = Memory_contiguous_row_major<float, 1>;
      using value_type = memory_type::value_type;

      for (size_t n = 0; n < 2000; ++n)
      {
         srand(n);
         const auto size = generateUniformDistribution(10, 10000000);
         const auto init_value = generateUniformDistribution<value_type>(std::numeric_limits<value_type>::lowest(), std::numeric_limits<value_type>::max());
   
         memory_type memory({ size }, init_value);
         memory_type_cpu cpu({ size }, init_value);
         cudaMemcpy(cpu.at(0), memory.at(0), size * sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);

         for (size_t i = 0; i < size; ++i)
         {
            TESTER_ASSERT(*cpu.at(i) == init_value);
         }
      }
   }
};

TESTER_TEST_SUITE(TestMemoryCuda);
TESTER_TEST(testInit_cte);
TESTER_TEST_SUITE_END();