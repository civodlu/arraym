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
// Specific tests for CUDA as the CPU based tests are using deferencement
//

struct TestMemoryCuda
{
   void test_Conversion()
   {
      cuda_ptr<float> ptr(nullptr);
      float* value = ptr;
      TESTER_ASSERT(value == nullptr);
   }

   void testInit_cte()
   {
      using memory_type = Memory_cuda_contiguous_column_major<float, 1>;
      using memory_type_cpu = Memory_contiguous_row_major<float, 1>;
      using value_type = memory_type::value_type;
      
      for (size_t n = 0; n < 2000; ++n)
      {
         srand((int)n);
         const auto size = generateUniformDistribution(10, 10000000);
         const auto init_value = generateUniformDistribution<value_type>(std::numeric_limits<value_type>::lowest(), std::numeric_limits<value_type>::max());
   
         memory_type memory({ size }, init_value);
         memory_type_cpu cpu({ size }, init_value);
         cudaMemcpy(cpu.at(0), memory.at(0), size * sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);

         for (size_t i = 0; i < size; ++i)
         {
            const auto value = *cpu.at(i);
            TESTER_ASSERT(value == init_value);
         }
      }
   }

   void testAllocator()
   {
      static_assert(!is_allocator_gpu<std::allocator<float>>::value, "CPU allocator!");
      static_assert(is_allocator_gpu<AllocatorCuda<float>>::value, "CPU allocator!");
   }
   
   void testDeepCopy()
   {
      using memory_type = Memory_cuda_contiguous_column_major<float, 1>;
      using memory_type_cpu = Memory_contiguous_row_major<float, 1>;
      using value_type = memory_type::value_type;

      static_assert(std::is_same<memory_type::const_pointer_type, const cuda_ptr<float>>::value, "must be a pointer to const float!");

      memory_type memory({ 5 }, -1.0f);

      memory_type memory2 = memory;
      memory_type_cpu cpu(memory.shape());
      cudaMemcpy(cpu.at(0), memory2.at(0), memory2.shape()[0] * sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);
      TESTER_ASSERT(*cpu.at(0) == -1.0f);
      TESTER_ASSERT(*cpu.at(1) == -1.0f);
      TESTER_ASSERT(*cpu.at(2) == -1.0f);
      TESTER_ASSERT(*cpu.at(3) == -1.0f);
      TESTER_ASSERT(*cpu.at(4) == -1.0f);
   }

   void testCallable()
   {
      struct Op
      {
         float operator()(cuda_ptr<float>) const
         {
            return 0.0f;
         }
      };

      static_assert(is_callable_with<Op, cuda_ptr<float>>::value, "implicit type conversion!");

      struct ConstOp
      {
         float operator()(const cuda_ptr<float>) const
         {
            return 0.0f;
         }
      };

      static_assert(is_callable_with<ConstOp, cuda_ptr<float>>::value, "implicit type conversion!");
   }

   

   void testConstArrayCasts()
   {
      static_assert(std::is_same<const float*, array_add_const<float*>::type>::value, "must be the same");
      static_assert(std::is_same<const cuda_ptr<float>, array_add_const<cuda_ptr<float>>::type>::value, "must be the same");
      static_assert(std::is_same<const cuda_ptr<const float>, array_add_const<cuda_ptr<const float>>::type>::value, "must be the same");
      
      
      static_assert(std::is_same<float*, array_remove_const<const float*>::type>::value, "must be the same");
      static_assert(std::is_same<float*, array_remove_const<float*>::type>::value, "must be the same");
      static_assert(std::is_same<cuda_ptr<float>, array_remove_const<cuda_ptr<float>>::type>::value, "must be the same");
      static_assert(std::is_same<cuda_ptr<float>, array_remove_const<const cuda_ptr<float>>::type>::value, "must be the same");

      using Memory = const Memory_cuda_contiguous_column_major<float, 1>;
      using pointer_T = Memory::pointer_type;
      using pointer_const_T = array_add_const<pointer_T>::type;
      using processor_T = ConstMemoryProcessor_contiguous_byMemoryLocality<Memory>;

      std::cout << typeid( pointer_const_T ).name() << std::endl;
      std::cout << typeid( processor_T::const_pointer_type ).name() << std::endl;
      std::cout << typeid( Memory::const_pointer_type ).name() << std::endl;
      
      static_assert( std::is_same<pointer_const_T, processor_T::const_pointer_type>::value, "must be the same!" );
   }

   void testCpuGpuTransfers_operatorequal()
   {
      using MemoryCpu = Memory_contiguous_column_major<float, 2>;
      using MemoryGpu = Memory_cuda_contiguous_column_major<float, 2>;

      float values[] = { 1, 2, 3, 4, 5, 6 };
      MemoryCpu mem_cpu1({ 2, 3 }, values);
      MemoryGpu mem_gpu;
      mem_gpu = mem_cpu1;

      MemoryCpu mem_cpu2;
      mem_cpu2 = mem_gpu;
      
      TESTER_ASSERT(*mem_cpu1.at({ 0, 0 }) == *mem_cpu2.at({ 0, 0 }));
      TESTER_ASSERT(*mem_cpu1.at({ 0, 1 }) == *mem_cpu2.at({ 0, 1 }));
      TESTER_ASSERT(*mem_cpu1.at({ 0, 2 }) == *mem_cpu2.at({ 0, 2 }));

      TESTER_ASSERT(*mem_cpu1.at({ 1, 0 }) == *mem_cpu2.at({ 1, 0 }));
      TESTER_ASSERT(*mem_cpu1.at({ 1, 1 }) == *mem_cpu2.at({ 1, 1 }));
      TESTER_ASSERT(*mem_cpu1.at({ 1, 2 }) == *mem_cpu2.at({ 1, 2 }));
   }

   void testCpuGpuTransfers_copyconstructor()
   {
      using MemoryCpu = Memory_contiguous_column_major<float, 2>;
      using MemoryGpu = Memory_cuda_contiguous_column_major<float, 2>;

      float values[] = { 1, 2, 3, 4, 5, 6 };
      MemoryCpu mem_cpu1({ 2, 3 }, values);
      MemoryGpu mem_gpu = mem_cpu1;

      MemoryCpu mem_cpu2 = mem_gpu;

      TESTER_ASSERT(*mem_cpu1.at({ 0, 0 }) == *mem_cpu2.at({ 0, 0 }));
      TESTER_ASSERT(*mem_cpu1.at({ 0, 1 }) == *mem_cpu2.at({ 0, 1 }));
      TESTER_ASSERT(*mem_cpu1.at({ 0, 2 }) == *mem_cpu2.at({ 0, 2 }));

      TESTER_ASSERT(*mem_cpu1.at({ 1, 0 }) == *mem_cpu2.at({ 1, 0 }));
      TESTER_ASSERT(*mem_cpu1.at({ 1, 1 }) == *mem_cpu2.at({ 1, 1 }));
      TESTER_ASSERT(*mem_cpu1.at({ 1, 2 }) == *mem_cpu2.at({ 1, 2 }));
   }

   void testSlicing()
   {
      using MemoryCpu = Memory_contiguous_column_major<float, 2>;
      using MemoryGpu = Memory_cuda_contiguous_column_major<float, 2>;

      float values[] = { 1, 2, 3, 4, 5, 6 };
      MemoryCpu mem_cpu1({ 2, 3 }, values);
      MemoryGpu mem_gpu;
      mem_gpu = mem_cpu1;

      {
         auto mem_gpu2 = mem_gpu.slice<0>({ 1, 0 });
         Memory_contiguous_column_major<float, 1> mem_cpu2;
         mem_cpu2 = mem_gpu2;
         TESTER_ASSERT(mem_cpu2.shape() == vector1ui{ 3 });
         TESTER_ASSERT(*mem_cpu2.at({ 0 }) == 4);
         TESTER_ASSERT(*mem_cpu2.at({ 1 }) == 5);
         TESTER_ASSERT(*mem_cpu2.at({ 2 }) == 6);
      }

      {
         auto mem_gpu2 = mem_gpu.slice<1>({ 0, 1 });
         Memory_contiguous_column_major<float, 1> mem_cpu2;
         mem_cpu2 = mem_gpu2;
         TESTER_ASSERT(mem_cpu2.shape() == vector1ui{ 2 });
         TESTER_ASSERT(*mem_cpu2.at({ 0 }) == 2);
         TESTER_ASSERT(*mem_cpu2.at({ 1 }) == 5);
      }
   }

   void testStrides()
   {
      using MemoryCpu = Memory_contiguous_column_major<float, 2>;
      using MemoryGpu = Memory_cuda_contiguous_column_major<float, 2>;

      float values[] = { 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0};
      MemoryCpu mem_cpu1({ 2, 3 }, values, vector2ui(2, 6));
      TESTER_ASSERT(*mem_cpu1.at({ 0, 0 }) == 1);
      TESTER_ASSERT(*mem_cpu1.at({ 1, 0 }) == 2);
      TESTER_ASSERT(*mem_cpu1.at({ 0, 1 }) == 4);
      TESTER_ASSERT(*mem_cpu1.at({ 1, 1 }) == 5);

      MemoryGpu mem_gpu;
      mem_gpu = mem_cpu1;

      MemoryCpu mem_cpu2 = mem_gpu;
      TESTER_ASSERT(*mem_cpu2.at({ 0, 0 }) == 1);
      TESTER_ASSERT(*mem_cpu2.at({ 1, 0 }) == 2);
      TESTER_ASSERT(*mem_cpu2.at({ 0, 1 }) == 4);
      TESTER_ASSERT(*mem_cpu2.at({ 1, 1 }) == 5);
   }

   void testStridedCopy()
   {
      float cpu_1[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
      auto gpu_1 = cuda::allocate_gpu<float>(10);
      auto gpu_2 = cuda::allocate_gpu<float>(10);
      cuda::kernel_copy(cpu_1, 2, cuda_ptr<float>(gpu_1.get()), 1, 5);

      float cpu_2[10];
      cuda::kernel_copy(cuda_ptr<float>(gpu_1.get()), 1, cpu_2, 1, 5);
      TESTER_ASSERT(cpu_2[0] == 1);
      TESTER_ASSERT(cpu_2[1] == 3);
      TESTER_ASSERT(cpu_2[2] == 5);
      TESTER_ASSERT(cpu_2[3] == 7);
      TESTER_ASSERT(cpu_2[4] == 9);

      float cpu_3[10];
      cuda::kernel_copy(cuda_ptr<float>(gpu_1.get()), 1, cpu_3, 2, 5);
      TESTER_ASSERT(cpu_3[0] == 1);
      TESTER_ASSERT(cpu_3[2] == 3);
      TESTER_ASSERT(cpu_3[4] == 5);
      TESTER_ASSERT(cpu_3[6] == 7);
      TESTER_ASSERT(cpu_3[8] == 9);

      // device to device
      float cpu_4[10];
      cuda::kernel_copy(cuda_ptr<float>(gpu_1.get()), 2, cuda_ptr<float>(gpu_2.get()), 1, 2);
      cuda::kernel_copy(cuda_ptr<float>(gpu_2.get()), 1, cpu_4, 1, 2);
      TESTER_ASSERT(cpu_4[0] == 1);
      TESTER_ASSERT(cpu_4[1] == 5);
   }
};

TESTER_TEST_SUITE(TestMemoryCuda);
TESTER_TEST(testAllocator);
TESTER_TEST(testInit_cte);
TESTER_TEST(test_Conversion);
TESTER_TEST(testDeepCopy);
TESTER_TEST(testConstArrayCasts);
TESTER_TEST(testCpuGpuTransfers_operatorequal);
TESTER_TEST(testCpuGpuTransfers_copyconstructor);
TESTER_TEST(testSlicing);
TESTER_TEST(testStrides);
TESTER_TEST(testStridedCopy);
TESTER_TEST_SUITE_END();

#endif