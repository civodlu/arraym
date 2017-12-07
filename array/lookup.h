#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Look up the array's element given a list of indexes

array = [4, 3, 2, 5, 1]
lookup(array, [1, 3]) = [3, 5]
*/
template <class T, size_t N, class Config>
Array<T, 1, typename Config::template rebind_dim<1>::other> lookup(const Array<T, N, Config>& array, const std::vector<StaticVector<ui32, N>>& indexes)
{
   using result_type = Array<T, 1, typename Config::template rebind_dim<1>::other>;

   result_type r(indexes.size());
   for (auto index : range(indexes))
   {
      r(index) = array(indexes[index]);
   }
   return r;
}

/**
@brief Look up the array's element given a list of indexes

array = [4, 3, 2, 5, 1]
lookup(array, [1, 3]) = [3, 5]
*/
template <class T, size_t N, class Config, class Config2>
Array<T, 1, typename Config::template rebind_dim<1>::other> lookup( const Array<T, N, Config>& array, const Array<StaticVector<ui32, N>, 1, Config2>& indexes )
{
   using result_type = Array<T, 1, typename Config::template rebind_dim<1>::other>;

   result_type r(indexes.shape()[0]);
   for (ui32 index = 0; index < indexes.shape()[0]; ++index)
   {
      r( vector1ui{ index } ) = array( indexes( vector1ui{ index } ) );
   }
   return r;
}

DECLARE_NAMESPACE_NLL_END
