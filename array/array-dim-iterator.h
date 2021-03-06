#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
template <class ArrayT, ui32 dim>
struct ArrayDimIterator
{
public:
   using array_type     = ArrayT;
   using array_ref_type = ArrayRef<typename ArrayT::value_type, ArrayT::RANK, typename ArrayT::Config>;
   using index_type     = typename array_type::index_type;
   using value_type     = ArrayT;

   using reference       = array_ref_type;
   using const_reference = const array_ref_type;

   ArrayDimIterator(array_type* array, ui32 current) : _array(array), _current(current)
   {
   }

   ArrayDimIterator& operator++()
   {
      ++_current;
      return *this;
   }

   bool operator==(const ArrayDimIterator& other) const
   {
      NLL_FAST_ASSERT(other._array == _array, "Must be based on the same array!");
      return other._current == _current;
   }

   bool operator!=(const ArrayDimIterator& other) const
   {
      return !operator==(other);
   }

   array_ref_type operator*()
   {
      index_type min_index;
      index_type max_index = _array->shape() - 1;
      min_index[dim]       = _current;
      max_index[dim]       = _current;

      using unconst_array_type = typename std::remove_cv<array_type>::type;
      auto& array_uc           = const_cast<unconst_array_type&>(*_array);
      return array_uc(min_index, max_index);
   }

private:
   array_type* _array;
   ui32 _current;
};

template <class ArrayT, ui32 dim>
struct ArrayDimIterator_proxy
{
   using iterator       = ArrayDimIterator<ArrayT, dim>;
   using const_iterator = ArrayDimIterator<ArrayT, dim>; // TODO const

   ArrayDimIterator_proxy(iterator begin, iterator end) : _begin(begin), _end(end)
   {
   }

   iterator begin() const
   {
      return _begin;
   }

   iterator end() const
   {
      return _end;
   }

private:
   iterator _begin;
   iterator _end;
};

template <class Proxy, class value_type>
struct ArrayValueIterator /*: std::iterator<std::input_iterator_tag, value_type>*/
{
   ArrayValueIterator(Proxy* proxy) : _proxy(proxy)
   {
      // point to the first element
      if (_proxy)
      {
         _proxy->accessSingleElement(_ptr);
         _nbElements = static_cast<ui32>(_proxy->_array.size());
      }
   }

   ArrayValueIterator& operator++()
   {
      _proxy->accessSingleElement(_ptr);
      ++_nbElementRead;
      return *this;
   }

   bool operator==(const ArrayValueIterator& other) const
   {
      (void*)&other; // remove unused warning
      NLL_FAST_ASSERT(other._proxy == nullptr, "MUST be NULL, we are going through all the elements of the array");
      return _nbElementRead == _nbElements;
   }

   bool operator!=(const ArrayValueIterator& other) const
   {
      return !operator==(other);
   }

   value_type& operator*()
   {
      return *_ptr;
   }

private:
   Proxy*       _proxy = nullptr;
   value_type*  _ptr = nullptr;
   ui32         _nbElementRead = 0;
   ui32         _nbElements = 0;
};

// this class is only designed to be called in the for(auto value : values(array))  (i.e., we go through all
// the elements of the array in order
template <class ArrayT>
struct ArrayValueIterator_proxy
{
   using value_type = typename ArrayT::value_type;
   using processor_type = ArrayProcessor_contiguous_byMemoryLocality<ArrayT>;
   using iterator = ArrayValueIterator<processor_type, value_type>;
   using const_iterator = ArrayValueIterator<processor_type, const value_type>;

   ArrayValueIterator_proxy(ArrayT& array) : _processor(array, 1) // here we operate element by element
   {}

   ~ArrayValueIterator_proxy()
   {}

   iterator begin()
   {
      return iterator(&_processor);
   }

   /*
   const_iterator begin() const
   {
      return const_iterator(&_processor);
   }*/

   iterator end()
   {
      return iterator(nullptr);  // this is a fake "end" as the processor knows when to stop
   }

   /*
   const_iterator end() const
   {
      return const_iterator(nullptr);
   }*/

private:
   processor_type _processor;
};
}

/**
 @brief iterate over the rows of an array. Each row will have the same number of dimensions as the array
 */
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 1> rows(Array<T, N, Config>& array)
{
   static_assert(N >= 2, "must have at least 2 dimensions!");
   static const ui32 dim = 1;
   using proxy_type      = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the rows of an array. Each row will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 1> rows(const Array<T, N, Config>& array)
{
   // TODO here we lose constness... (copy a reference)
   static_assert(N >= 2, "must have at least 2 dimensions!");
   static const ui32 dim = 1;
   using proxy_type      = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::const_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the rows of an array. Each row will have the same number of dimensions as the array

Matrices must be handled separately due to the storage format
*/
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
details::ArrayDimIterator_proxy<Matrix<T, Mapper, Allocator>, 0> rows(Matrix<T, Mapper, Allocator>& array)
{
   static const ui32 dim = 0;
   using proxy_type      = details::ArrayDimIterator_proxy<Matrix<T, Mapper, Allocator>, dim>;
   using iter_type       = typename proxy_type::iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the rows of an array. Each row will have the same number of dimensions as the array

Matrices must be handled separately due to the storage format
*/
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
details::ArrayDimIterator_proxy<const Matrix<T, Mapper, Allocator>, 0> rows(const Matrix<T, Mapper, Allocator>& array)
{
   static const ui32 dim = 0;
   using proxy_type      = details::ArrayDimIterator_proxy<const Matrix<T, Mapper, Allocator>, dim>;
   using iter_type       = typename proxy_type::const_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 0> columns(Array<T, N, Config>& array)
{
   static_assert(N >= 2, "must have at least 2 dimensions!");
   static const ui32 dim = 0;
   using proxy_type      = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 0> columns(const Array<T, N, Config>& array)
{
   // TODO here we lose constness... (copy a reference)
   static_assert(N >= 2, "must have at least 2 dimensions!");
   static const ui32 dim = 0;
   using proxy_type      = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::const_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array

Matrices must be handled separately due to the storage format
*/
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
details::ArrayDimIterator_proxy<Matrix<T, Mapper, Allocator>, 1> columns(Matrix<T, Mapper, Allocator>& array)
{
   static const ui32 dim = 1;
   using proxy_type      = details::ArrayDimIterator_proxy<Matrix<T, Mapper, Allocator>, dim>;
   using iter_type       = typename proxy_type::iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array

Matrices must be handled separately due to the storage format
*/
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
details::ArrayDimIterator_proxy<const Matrix<T, Mapper, Allocator>, 1> columns(const Matrix<T, Mapper, Allocator>& array)
{
   static const ui32 dim = 1;
   using proxy_type      = details::ArrayDimIterator_proxy<const Matrix<T, Mapper, Allocator>, dim>;
   using iter_type       = typename proxy_type::const_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the slices of an array. Each slice will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 2> slices(const Array<T, N, Config>& array)
{
   // TODO here we lose constness... (copy a reference)
   static_assert(N >= 3, "must have at least 3 dimensions!");
   static const ui32 dim = 2;
   using proxy_type      = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::const_iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief iterate over the slices of an array. Each slice will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 2> slices(Array<T, N, Config>& array)
{
   static_assert(N >= 3, "must have at least 3 dimensions!");
   static const ui32 dim = 2;
   using proxy_type      = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type       = typename proxy_type::iterator;
   return proxy_type(iter_type(&array, 0), iter_type(&array, array.shape()[dim]));
}

/**
@brief Iterate on the values of an array
*/
template <class T, size_t N, class Config>
details::ArrayValueIterator_proxy<Array<T, N, Config>> values(Array<T, N, Config>& array)
{
   return details::ArrayValueIterator_proxy<Array<T, N, Config>>(array);
}

/**
 @brief Iterate on the values of an array
 */
template <class T, size_t N, class Config>
details::ArrayValueIterator_proxy<Array<T, N, Config>> values(const Array<T, N, Config>& array)
{
   // TODO losing constness here
   // placeholder implementation. Needs to be properly implemented!
   // We MUST revisit all these iterators: losing constness, non-const->const iterator conversion... 
   return details::ArrayValueIterator_proxy<Array<T, N, Config>>(const_cast<Array<T, N, Config>&>(array));
}

/**
 @brief return a reference of a column of a matrix
 */
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
typename Matrix<T, Mapper, Allocator>::array_type_ref column(const Matrix<T, Mapper, Allocator>& array_c, ui32 col)
{
   using array_type = Matrix<T, Mapper, Allocator>;
   using array_ref_type = ArrayRef<T, 2, typename array_type::Config>;
 
   auto& array = const_cast<array_type&>(array_c); // TODO losing constness here. Must revisit the semantics of const arrays

   StaticVector<ui32, 2> max = array.shape() - 1;
   max[1] = col;
   StaticVector<ui32, 2> min;
   min[1] = col;

   return array(min, max);
}

/**
@brief return a reference of a row of a matrix
*/
template <class T, class Mapper, class Allocator, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
typename Matrix<T, Mapper, Allocator>::array_type_ref row(const Matrix<T, Mapper, Allocator>& array_c, ui32 row)
{
   using array_type = Matrix<T, Mapper, Allocator>;
   using array_ref_type = ArrayRef<T, 2, typename array_type::Config>;

   auto& array = const_cast<array_type&>(array_c); // TODO losing constness here. Must revisit the semantics of const arrays

   StaticVector<ui32, 2> max = array.shape() - 1;
   max[0] = row;
   StaticVector<ui32, 2> min;
   min[0] = row;

   return array(min, max);
}

/**
@brief copy rows of a matrix
*/
template <class T, class Mapper, class Allocator, class T2, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
Matrix<T, Mapper, Allocator> rows_copy(const Matrix<T, Mapper, Allocator>& array, const std::vector<T2>& row_ids)
{
   Matrix<T, Mapper, Allocator> cpy(row_ids.size(), array.columns());
   for (auto it : enumerate(row_ids))
   {
      typename Matrix<T, Mapper, Allocator>::array_type_ref ref_left = row(cpy, it.index);
      typename Matrix<T, Mapper, Allocator>::array_type_ref ref_right = row(array, *it);

      ref_left = ref_right;
   }
   return cpy;
}

/**
@brief copy columns from a matrix
*/
template <class T, class Mapper, class Allocator, class T2, typename = typename std::enable_if<is_matrix<Matrix<T, Mapper, Allocator>>::value>::type>
Matrix<T, Mapper, Allocator> columns_copy(const Matrix<T, Mapper, Allocator>& array, const std::vector<T2>& col_ids)
{
   Matrix<T, Mapper, Allocator> cpy(array.rows(), col_ids.size());
   for (auto it : enumerate(col_ids))
   {
      typename Matrix<T, Mapper, Allocator>::array_type_ref ref_left = column(cpy, it.index);
      typename Matrix<T, Mapper, Allocator>::array_type_ref ref_right = column(array, *it);

      ref_left = ref_right;
   }
   return cpy;
}

/**
@brief return a reference of a row of an array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
typename Array<T, N, Config>::array_type_ref row(const Array<T, N, Config>& array_c, ui32 row)
{
   static_assert(N >= 2, "must have at leas rank 2!");
   using array_type = Array< T, N, Config>;
   using array_ref_type = typename array_type::array_type_ref;

   auto& array = const_cast<array_type&>(array_c); // TODO losing constness here. Must revisit the semantics of const arrays

   StaticVector<ui32, N> max = array.shape() - 1;
   max[1] = row;
   StaticVector<ui32, N> min;
   min[1] = row;

   return array(min, max);
}

/**
@brief return a reference of a column of an array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
typename Array<T, N, Config>::array_type_ref column(const Array<T, N, Config>& array_c, ui32 col)
{
   using array_type = Array< T, N, Config>;
   using array_ref_type = typename array_type::array_type_ref;

   auto& array = const_cast<array_type&>(array_c); // TODO losing constness here. Must revisit the semantics of const arrays

   StaticVector<ui32, N> max = array.shape() - 1;
   max[0] = col;
   StaticVector<ui32, N> min;
   min[0] = col;

   return array(min, max);
}

/**
@brief return a reference of a slice of an array
*/
template <class T, size_t N, class Config, typename = typename std::enable_if<!is_matrix<Array<T, N, Config>>::value>::type>
typename Array<T, N, Config>::array_type_ref slice(const Array<T, N, Config>& array_c, ui32 slice)
{
   static_assert(N >= 3, "must have at leas rank 2!");
   using array_type = Array< T, N, Config>;
   using array_ref_type = typename array_type::array_type_ref;

   auto& array = const_cast<array_type&>(array_c); // TODO losing constness here. Must revisit the semantics of const arrays

   StaticVector<ui32, N> max = array.shape() - 1;
   max[2] = slice;
   StaticVector<ui32, N> min;
   min[2] = slice;

   return array(min, max);
}

DECLARE_NAMESPACE_NLL_END
