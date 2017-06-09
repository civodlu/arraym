#pragma once

DECLARE_NAMESPACE_NLL

/**
 @brief Transpose a matrix
 */
template <class T, class Config>
Array<T, 2, Config> transpose(const Array<T, 2, Config>& m)
{
   // TODO faster implementation
   Array<T, 2, Config> r({ m.shape()[1], m.shape()[0] });
   for (size_t nx = 0; nx < r.shape()[1]; ++nx)
   {
      for (size_t ny = 0; ny < r.shape()[0]; ++ny)
      {
         r(ny, nx) = m(nx, ny);
      }
   }
   return r;
}

/**
 @brief Transpose for higher dimensional array
 @param new_axis the order of the axis of the returned matrix
 */
template <class T, size_t N, class Config>
Array<T, N, Config> transpose(const Array<T, N, Config>& m, const StaticVector<ui32, N>& new_axis)
{
   using index_type = StaticVector<ui32, N>;
   using array_result = Array<T, N, Config>;
   using const_pointer_type = typename array_result::const_pointer_type;

   auto reorder = [&](const index_type& i)
   {
      index_type ir(no_init_tag);
      for (size_t n = 0; n < N; ++n)
      {
         ir[n] =i[new_axis[n]];
      }
      return ir;
   };

   const index_type new_shape = reorder(m.shape());
   Array<T, N, Config> result(new_shape);
   bool hasMoreElements = true;
   ConstArrayProcessor_contiguous_byMemoryLocality<array_result> it_result(m, 0);
   const auto varying_index = it_result.getVaryingIndex();
   const auto dimension_to_copy = new_axis[varying_index];
   while (hasMoreElements)
   {
      // read a full memory line
      const_pointer_type ptr(nullptr);
      hasMoreElements = it_result.accessMaxElements(ptr);
      const auto currentIndex = it_result.getArrayIndex();

      // get the subarray's result for the corresponding line
      index_type min_index = reorder(currentIndex);
      index_type max_index = min_index;
      min_index[dimension_to_copy] = 0;
      max_index[dimension_to_copy] = m.shape()[varying_index] - 1;
      auto sub = result(min_index, max_index);

      // TODO only works for contiguous (not slice based arrays!)
      const auto& stride_sub = sub.getMemory().getIndexMapper()._getPhysicalStrides();
      auto stride = stride_sub[dimension_to_copy];
      details::copy_naive(array_base_memory(sub), stride, ptr, 1, sub.size());
   }

   return result;
}

DECLARE_NAMESPACE_NLL_END
