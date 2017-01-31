#pragma once

#ifdef WITH_CUDA

DECLARE_NAMESPACE_NLL

template <class T, size_t N, class Allocator>
Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>> cos(const Array<T, N, details::ArrayTraitsConfigCuda<T, N, Allocator>>& array);

DECLARE_NAMESPACE_NLL_END

#endif // WITH_CUDA