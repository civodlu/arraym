#include "cuda-utils.h"

#ifdef WITH_CUDA

#include <cuda_runtime.h>

DECLARE_NAMESPACE_NLL

namespace cuda
{
   void _cudaAssert(int code, const char *file, int line)
   {
      if (cudaError_t(code) != cudaError::cudaSuccess)
      {
         std::cerr << "GPUassert:" << cudaGetErrorString(cudaError_t(code)) << " file=" << file << " line=" << line << std::endl;
         exit(code);
      }
   }

   void _cudaCheck(const char *file, int line)
   {
      _cudaAssert(cudaPeekAtLastError(), file, line);
   }
}

DECLARE_NAMESPACE_NLL_END

#endif