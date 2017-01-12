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
   void testInit()
   {
      float* ptr;
      const size_t size = 2000;
      
      CHECK_CUDA(cudaMalloc<float>(&ptr, size * sizeof(float)));
      //::cuda::init_kernel2<float>(ptr, -4.0f, 2000);
      NAMESPACE_NLL::cuda::init_kernel(ptr, -4.0f, size);
      NAMESPACE_NLL::cuda::cudaCheck();
      
      std::vector<float> cpu(size*2);
      /*
      for (auto& v : cpu)
      {
         v = -4.0f;
      }*/

      cudaMemcpy(&cpu[0], ptr, size * sizeof(float), cudaMemcpyKind::cudaMemcpyDeviceToHost);

      std::cout << cpu[1999] << std::endl;

      //NAMESPACE_NLL::cuda::init_kernel2(ptr, -4.0f, 2000);

      //::cuda::init_kernel(ptr, -4.0f, 2000);
      //Memory_gpu_cuda<float, 1> memory({ 1024 }, -42.0f);
   }

   void testInit2()
   {
      testInit();
   }
};

TESTER_TEST_SUITE(TestMemoryCuda);
TESTER_TEST(testInit); 
TESTER_TEST(testInit2);
TESTER_TEST_SUITE_END();