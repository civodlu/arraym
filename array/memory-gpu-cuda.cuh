#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace cuda
{
   template<typename T>
   void init_kernel(T* ptr, const T val, const size_t nb_elements);
}

DECLARE_NAMESPACE_NLL_END

#endif