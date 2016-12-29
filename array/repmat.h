#pragma once

DECLARE_NAMESPACE_NLL

template <class T, size_t N, class Config, size_t N2>
Array<T, N2, typename Config::template rebind_dim<N2>::other> repmat(const Array<T, N, Config>& array, const StaticVector<ui32, N2>& times)
{
   static_assert(N2 >= N, "N2 must have at least the same dimension as the array");
   using other_array_type = Array<T, N2, typename Config::template rebind_dim<N2>::other>;
   using array_type = Array<T, N, Config>;

   // first convert to the correct dimention
   typename other_array_type::index_type shape_n2;
   typename other_array_type::index_type other_shape;
   for (size_t n = 0; n < N; ++n)
   {
      shape_n2[n] = array.shape()[n];
      other_shape[n] = array.shape()[n] * times[n];
   }
   for (size_t n = N; n < N2; ++n)
   {
      shape_n2[n] = 1;
      other_shape[n] = times[n];
   }
   const auto array_n2 = as_array(array, shape_n2);

   // then iterate array in a memory friendly manner
   other_array_type other(other_shape);
   ArrayChunking_contiguous_base<other_array_type> chunking(times, getFastestVaryingIndexes(other));

   bool more_elements = true;
   while (more_elements)
   {
      const auto index = chunking.getArrayIndex() * shape_n2;
      more_elements = chunking._accessElements(1);
      other(index, index + shape_n2 - 1) = array_n2;
   }

   return other;
}

DECLARE_NAMESPACE_NLL_END