#pragma once

#include <array/config.h>
#include <array/array-api.h>

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace cuda
{
   ARRAY_API void _cudaAssert(int code, const char *file, int line);
   ARRAY_API void _cudaCheck(const char *file, int line);
}

/// assert there was no CUDA error
#define cudaAssert(ans) _cudaAssert((ans), __FILE__, __LINE__)

/// assert there was no error during kernel execution
#define cudaCheck() _cudaCheck(__FILE__, __LINE__)

DECLARE_NAMESPACE_NLL_END

#endif