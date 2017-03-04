#pragma once

DECLARE_NAMESPACE_NLL

/**
 @brief Concat two array on a given axis


                 | 1, 2, 3 |                                            | 1, 2, 3 |
 concat array1 = | 4, 5, 6 | array2 = | 7, 8, 9 |, axis = 1    result = | 4, 5, 6 |
                                                                        | 7, 8, 9 |
 
                | 1, 4 |          | 7 |                          | 1, 4, 7 |
 concat arra1 = | 2, 5 | array2 = | 8 |, axis = 0,      result = | 2, 5, 8 |
                | 3, 6 |          | 9 |                          | 3, 6, 9 |
 */
template <class T, size_t N, class Config>
Array<T, N, Config> concat(const Array<T, N, Config>& array1, const Array<T, N, Config>& array2, size_t axis)
{
   using array_type = Array<T, N, Config>;
   using index_type = typename array_type::index_type;

   for (auto index : range(N))
   {
      if (index != axis)
      {
         ensure(array1.shape()[index] == array2.shape()[index], "arrays must have the same dimension for the joint");
      }
   }

   auto shape  = array1.shape();
   shape[axis] = array1.shape()[axis] + array2.shape()[axis];
   Array<T, N, Config> r(shape);

   index_type min_index;
   index_type max_index = array1.shape() - 1;
   r(min_index, max_index) = array1;

   min_index[axis] += array1.shape()[axis];
   max_index = array2.shape() - 1;
   max_index[axis] += array1.shape()[axis];
   r(min_index, max_index) = array2;
   return r;
}

DECLARE_NAMESPACE_NLL_END
