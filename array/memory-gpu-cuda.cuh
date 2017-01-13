#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

namespace cuda
{
   template<typename T>
   void kernel_init(T* ptr, const T val, const size_t nb_elements);

   template<typename T>
   void kernel_copy(const T* input, T* output, const size_t nb_elements);
}

DECLARE_NAMESPACE_NLL_END

#endif