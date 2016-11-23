#pragma once

DECLARE_NAMESPACE_NLL

/**
 @file

 This file defines "processors", which allow to iterate over an array or multiple arrays in the most efficient manner considering
 the memory locality of the arrays
 */

namespace details
{
   /**
   @brief Abstract the array traversal. This is based on knwown structure of Array::Memory

   @tparam Array an array

   The only assumption for this iterator is that we have the fastest varying index has at
   least <maxAccessElements> contiguous memory elements. So this handle <Memory_contiguous>
   and <Memory_multislice> Memories

   @todo if fastest varying dimension size = 1 -> skip this dimension for contiguous memory only
   */
   template <class Array>
   class ArrayProcessor_contiguous_base
   {
      using diterator = typename Array::diterator;
      static_assert( std::is_base_of<memory_layout_linear, typename Array::Memory>::value, "must be a linear index mapper!" );

   public:
      using index_type = typename Array::index_type;
      using pointer_type = typename Array::pointer_type;

      template <class FunctorGetDimensionOrder>
      ArrayProcessor_contiguous_base( Array& array, const FunctorGetDimensionOrder& functor ) :
         _array( array )
      {
         _indexesOrder = functor( array );

         for ( ui32 n = 0; n < _indexesOrder.size(); ++n )
         {
            _indexesOrderInv[ _indexesOrder[ n ] ] = n;
            _sizeOrder[ n ] = array.shape()[ _indexesOrder[ n ] ];
         }
         _maxAccessElements = array.shape()[ _indexesOrder[ 0 ] ];
      }

      bool accessSingleElement( pointer_type& ptrToValue )
      {
         return _accessElements( ptrToValue, 1 );
      }

      // this is the specific view index reordered by <functor>
      const index_type& getIteratorIndex() const
      {
         return _iterator_index;
      }

      const index_type getArrayIndex() const
      {
         index_type indexReordered;  // start with min offset
         for ( size_t n = 0; n < Array::RANK; ++n )
         {
            // currently the index is expressed from fastest to lowest varying speed so transform it back
            indexReordered[ n ] += _iterator_index[ _indexesOrderInv[ n ] ];
         }
         return indexReordered;
      }

      ui32 getVaryingIndex() const
      {
         return _indexesOrder[ 0 ];
      }

      const index_type& getVaryingIndexOrder() const
      {
         return _indexesOrder;
      }

   protected:
      bool _accessElements( pointer_type& ptrToValue, ui32 nbElements )
      {
         if ( _pointer_invalid )
         {
            _iterator = _array.beginDim( _indexesOrder[ 0 ], getArrayIndex() );
            _pointer_invalid = false;
         } else
         {
            _iterator.add( nbElements );
         }
         ptrToValue = &( *_iterator );

         const bool hasMoreElements = Increment<0, false>::run( _iterator_index, _sizeOrder, _pointer_invalid, nbElements );
         return hasMoreElements;
      }

      template <int I, bool B>
      struct Increment
      {
         FORCE_INLINE static bool run( core::StaticVector<ui32, Array::RANK>& index, const core::StaticVector<ui32, Array::RANK>& size, bool& recomputeIterator, ui32 nbElements )
         {
            index[ I ] += nbElements;
            if ( index[ I ] == size[ I ] )
            {
               recomputeIterator = true;
               for ( size_t n = 0; n <= I; ++n )
               {
                  index[ n ] = 0;
               }
               return Increment<I + 1, ( I + 1 ) == Array::RANK>::run( index, size, recomputeIterator, 1 );
            }
            return true;
         }
      };

      template <int I>
      struct Increment < I, true >
      {
         FORCE_INLINE static bool run( core::StaticVector<ui32, Array::RANK>&, const core::StaticVector<ui32, Array::RANK>&, bool&, ui32 )
         {
            return false;
         }
      };

   protected:
      Array&                           _array;
      diterator                        _iterator;
      bool                             _pointer_invalid = true;
      index_type                       _iterator_index;

      index_type                       _sizeOrder;         // the size, ordered by <_indexesOrder>
      index_type                       _indexesOrder;      // the order of the traversal
      index_type                       _indexesOrderInv;   // the order of the traversal
      ui32                             _maxAccessElements; // maximum number of steps in the fastest varying dimension possible without increasing the other indexes
   };

   template <class T, ui32 N, class ConfigT>
   core::StaticVector<ui32, N> getFastestVaryingIndexes( const Array<T, N, ConfigT>& array )
   {
      using array_type = Array<T, N, ConfigT>;
      using index_type = typename array_type::index_type;

      static_assert( std::is_base_of<memory_layout_linear, typename array_type::Memory>::value, "must be a linear index mapper!" );

      index_type fastestVaryingIndexes;

      // first, we want to iterate from the fastest->lowest varying index to avoid as much cache misses as possible
      // EXCEPT is stride is 0, which is a special case (different slices in memory, so this is actually the WORST dimension to iterate on)
      auto strides = array.getMemory().getIndexMapper()._getPhysicalStrides();
      for ( auto& v : strides )
      {
         if ( v == 0 )
         {
            v = std::numeric_limits<typename index_type::value_type>::max();
         }
      }

      std::array<std::pair<ui32, ui32>, N> stridesIndex;
      for ( ui32 n = 0; n < N; ++n )
      {
         stridesIndex[ n ] = std::make_pair( strides[ n ], n );
      }
      std::sort( stridesIndex.begin(), stridesIndex.end() );

      for ( ui32 n = 0; n < N; ++n )
      {
         fastestVaryingIndexes[ n ] = stridesIndex[ n ].second;
      }

      return fastestVaryingIndexes;
   }
}

/**
 @brief iterate an array by maximizing memory locality. This should be the preferred iterator
 */
template <class Array>
class ArrayProcessor_contiguous_byMemoryLocality : public details::ArrayProcessor_contiguous_base<Array>
{
public:
    using base = details::ArrayProcessor_contiguous_base<Array>;
    using pointer_type = typename base::pointer_type;

   ArrayProcessor_contiguous_byMemoryLocality( Array& array ) :
      base( array, &details::getFastestVaryingIndexes<typename Array::value_type, Array::RANK, typename Array::Config> )
   {}

   ui32 getMaxAccessElements() const
   {
      return this->_maxAccessElements;
   }

   ui32 stride() const
   {
      return this->_array.getMemory().getIndexMapper()._getPhysicalStrides()[ this->getVaryingIndex() ];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, <ptrToValue> if accessed in a contiguous fashion must account for the stride in the direction of access using <stride()>
   */
   bool accessMaxElements( pointer_type& ptrToValue )
   {
      return this->_accessElements( ptrToValue, this->_maxAccessElements );
   }
};

/**
@brief iterate by dimension, in (x, y, z...) order
*/
template <class Array>
class ArrayProcessor_contiguous_byDimension : public details::ArrayProcessor_contiguous_base < Array >
{
    using base = details::ArrayProcessor_contiguous_base < Array >;
    using index_type  = typename base::index_type;
   static index_type getIndexes( const Array& )
   {
      index_type indexes;
      for ( ui32 n = 0; n < Array::RANK; ++n )
      {
         indexes[ n ] = n;
      }
      return indexes;
   }

public:
   ArrayProcessor_contiguous_byDimension( Array& array ) :
      details::ArrayProcessor_contiguous_base<Array>( array, &getIndexes )
   {}
};

/**
@brief Generic fill of an array. The index order is defined by memory locality
@param functor will be called using functor(index_type(x, y, z, ...)), i.e., each coordinate components
*/
template <class T, int N, class Config, class Functor>
void fill( Array<T, N, Config>& array, Functor functor )
{
   using functor_return = typename function_traits<Functor>::return_type;
   static_assert( std::is_same<functor_return, T>::value, "functor return type must be the same as array type" );

   if ( array.isEmpty() )
   {
      return;
   }

   using array_type = Array < T, N, Config > ;
   bool hasMoreElements = true;

   ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator( array );
   while (hasMoreElements)
   {
      typename array_type::value_type* ptr = 0;
      const auto& currentIndex = iterator.getArrayIndex();
      hasMoreElements = iterator.accessSingleElement( ptr );
      *ptr = functor( currentIndex );
   }
}

DECLARE_NAMESPACE_END