#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector1ui = StaticVector<ui32, 1>;
using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

#include <cuda.h>
#include <cuda_runtime.h>

template <class T, size_t N, class Allocator = AllocatorCuda<T>>
using Memory_cuda_contiguous_column_major = Memory_contiguous<T, N, IndexMapper_contiguous_column_major<N>, Allocator, cuda_ptr<T>>;


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

      //Memory_cuda_contiguous_column_major
   }
   
   void testDeepCopy()
   {
      /*
      using memory_type = Memory_cuda_contiguous_column_major<float, 1>;
      using memory_type_cpu = Memory_contiguous_row_major<float, 1>;
      using value_type = memory_type::value_type;

      static_assert(std::is_same<memory_type::const_pointer_type, const cuda_ptr<float>>::value, "must be a pointer to const float!");

      memory_type memory({ 5 }, -1.0f);

      memory_type memory2 = memory;
      memory_type_cpu cpu(memory.shape());
      cudaMemcpy(cpu.at(0), memory.at(0), memory.shape()[0] * sizeof(value_type), cudaMemcpyKind::cudaMemcpyDeviceToHost);
      */
   }

   void testCallable()
   {
      struct Op
      {
         float operator()(cuda_ptr<float> p) const
         {
            return 0.0f;
         }
      };

      static_assert(is_callable_with<Op, cuda_ptr<float>>::value, "implicit type conversion!");

      struct ConstOp
      {
         float operator()(const cuda_ptr<float> p) const
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
      
      //static_assert( std::is_same<pointer_const_T, processor_T::const_pointer_type>::value, "must be the same!" );
   }
};

TESTER_TEST_SUITE(TestMemoryCuda);
//TESTER_TEST(testAllocator);
//TESTER_TEST(testInit_cte);
//TESTER_TEST(test_Conversion);
TESTER_TEST(testDeepCopy);
TESTER_TEST(testConstArrayCasts);
TESTER_TEST_SUITE_END();