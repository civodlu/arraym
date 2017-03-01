#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
   template <class ArrayT, ui32 dim>
   struct ArrayDimIterator
   {
   public:
      using array_type = ArrayT;
      using array_ref_type = ArrayRef<typename ArrayT::value_type, ArrayT::RANK, typename ArrayT::Config>;
      using index_type = typename array_type::index_type;

      ArrayDimIterator( array_type* array, ui32 current ) : _array( array ), _current( current )
      {}

      ArrayDimIterator& operator++( )
      {
         ++_current;
         return *this;
      }

      bool operator==( const ArrayDimIterator& other ) const
      {
         NLL_FAST_ASSERT( other._array == _array, "Must be based on the same array!" );
         return other._current == _current;
      }

      bool operator!=( const ArrayDimIterator& other ) const
      {
         return !operator==( other );
      }

      array_ref_type operator*( )
      {
         index_type min_index;
         index_type max_index = _array->shape() - 1;
         min_index[ dim ] = _current;
         max_index[ dim ] = _current;

         auto& array_uc = const_cast<std::remove_cv<typename array_type>::type&>( *_array );
         return array_uc( min_index, max_index );
      }

   private:
      array_type* _array;
      ui32        _current;
   };

   template <class ArrayT, ui32 dim>
   struct ArrayDimIterator_proxy
   {
      using array_iterator = ArrayDimIterator<ArrayT, dim>;

      ArrayDimIterator_proxy( array_iterator begin, array_iterator end ) : _begin( begin ), _end( end )
      {}

      array_iterator begin() const
      {
         return _begin;
      }

      array_iterator end() const
      {
         return _end;
      }

   private:
      array_iterator _begin;
      array_iterator _end;
   };
}

/**
 @brief iterate over the rows of an array. Each row will have the same number of dimensions as the array
 */
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 1> rows( Array<T, N, Config>& array )
{
   static_assert( N >= 2, "must have at least 2 dimensions!" );
   static const ui32 dim = 1;
   using proxy_type = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

/**
@brief iterate over the rows of an array. Each row will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 1> rows( const Array<T, N, Config>& array )
{
   // TODO here we lose constness... (copy a reference)
   static_assert( N >= 2, "must have at least 2 dimensions!" );
   static const ui32 dim = 1;
   using proxy_type = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 0> columns( Array<T, N, Config>& array )
{
   static_assert( N >= 2, "must have at least 2 dimensions!" );
   static const ui32 dim = 0;
   using proxy_type = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

/**
@brief iterate over the columns of an array. Each column will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 0> columns( const Array<T, N, Config>& array )
{
   // TODO here we lose constness... (copy a reference)
   static_assert( N >= 2, "must have at least 2 dimensions!" );
   static const ui32 dim = 0;
   using proxy_type = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

/**
@brief iterate over the slices of an array. Each slice will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<const Array<T, N, Config>, 2> slices( const Array<T, N, Config>& array )
{
   // TODO here we lose constness... (copy a reference)
   static_assert( N >= 3, "must have at least 3 dimensions!" );
   static const ui32 dim = 2;
   using proxy_type = details::ArrayDimIterator_proxy<const Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

/**
@brief iterate over the slices of an array. Each slice will have the same number of dimensions as the array
*/
template <class T, size_t N, class Config>
details::ArrayDimIterator_proxy<Array<T, N, Config>, 2> slices( Array<T, N, Config>& array )
{
   static_assert( N >= 3, "must have at least 3 dimensions!" );
   static const ui32 dim = 2;
   using proxy_type = details::ArrayDimIterator_proxy<Array<T, N, Config>, dim>;
   using iter_type = typename proxy_type::array_iterator;
   return proxy_type( iter_type( &array, 0 ), iter_type( &array, array.shape()[ dim ] ) );
}

DECLARE_NAMESPACE_NLL_END