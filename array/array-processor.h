#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Abstract the array traversal. This is based on knwown structure of Array::Memory

The only assumption for this iterator is that we have the fastest varying index has at
least <maxAccessElements> contiguous memory elements. So this handle <Memory_contiguous>
and <Memory_multislice> Memories
*/
template <class Array>
class ArrayIterator_contiguous_base
{
   using diterator = typename Array::diterator;
   static_assert( std::is_base_of<memory_layout_linear, typename Array::Memory>::value, "must be a linear index mapper!" );

public:
   using index_type = typename Array::index_type;
   using pointer_type = typename Array::pointer_type;

   template <class FunctorGetDimensionOrder>
   ArrayIterator_contiguous_base( Array& array, const FunctorGetDimensionOrder& functor ) :
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
      return _current_index;
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

namespace details
{
   template <class T, ui32 N, class ConfigT>
   core::StaticVector<ui32, N> getFastestVaryingIndexes( const Array<T, N, ConfigT>& array )
   {
      using array_type = Array<T, N, ConfigT>;
      using index_type = typename array_type::index_type;

      index_type fastestVaryingIndexes;

      // TODO static_assert(core::is_mapper_linear_base<typename array_type::IndexMapper>::value, "TODO only handled linear index mapper");

      // first, we want to iterate from the fastest->lowest varying index to avoid as much cache misses as possible
      // EXCEPT is stride is 0, which is a special case (different slices in memory, so this is actually the WORST dimension to iterate on)
      auto strides = array.getMemory().getIndexMapper()._getPhysicalStrides();
      for ( auto& v : strides )
      {
         if ( v == 0 )
         {
            v = std::numeric_limits<index_type::value_type>::max();
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

template <class Array>
class ArrayIterator_contiguous_byMemoryLocality : public ArrayIterator_contiguous_base<Array>
{
public:
   ArrayIterator_contiguous_byMemoryLocality( Array& array ) :
      ArrayIterator_contiguous_base<Array>( array, &details::getFastestVaryingIndexes<typename Array::value_type, Array::RANK, typename Array::Config> )
   {}

   ui32 getMaxAccessElements() const
   {
      return this->_maxAccessElements;
   }

   ui32 stride() const
   {
      return _array.getMemory().getIndexMapper()._getPhysicalStrides()[ getVaryingIndex() ];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, <ptrToValue> if accessed in a contiguous fashion must account for the stride in the direction of access using <stride()>
   */
   bool accessMaxElements( pointer_type& ptrToValue )
   {
      return this->_accessElements( ptrToValue, _maxAccessElements );
   }
};

DECLARE_NAMESPACE_END