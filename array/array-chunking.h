#pragma once

DECLARE_NAMESPACE_NLL

template <class Array>
class ConstMemoryProcessor_contiguous_byMemoryLocality;
template <class Memory>
class ConstArrayProcessor_contiguous_byMemoryLocality;

template <class Array>
class ArrayChunking_contiguous_base
{
   static_assert(std::is_base_of<memory_layout_linear, typename Array::Memory>::value, "must be a linear index mapper!");

   template <class Array2>
   friend class ConstMemoryProcessor_contiguous_byMemoryLocality;

   template <class Memory2>
   friend class ConstArrayProcessor_contiguous_byMemoryLocality;

public:
   using index_type = typename Array::index_type;

   /**
    @param nbElementsToAccessPerIter the number of elements that will be read for each call to @p _accessElements
                                     if nbElementsToAccessPerIter == 0, the number of elements will be set to the maximum possible
    */
   ArrayChunking_contiguous_base(const index_type& shape, const index_type& indexesOrder, ui32 nbElementsToAccessPerIter) : _shape(shape)
   {
      _indexesOrder    = indexesOrder;
      ui32 nb_elements = 1;
      for (ui32 n = 0; n < _indexesOrder.size(); ++n)
      {
         nb_elements *= shape[n];
         _indexesOrderInv[_indexesOrder[n]] = n;
         _sizeOrder[n]                      = shape[_indexesOrder[n]];
      }

      const auto maxAccessElements = shape[_indexesOrder[0]];
      if (nbElementsToAccessPerIter == 0)
      {
         _nbElementsToAccessPerIter = maxAccessElements;
      }
      else
      {
         _nbElementsToAccessPerIter = nbElementsToAccessPerIter;
      }
      ensure(_nbElementsToAccessPerIter == 1 || _nbElementsToAccessPerIter == maxAccessElements, "TODO handle different nbElements!");

      // after calling _maxNbAccess times @ref _accessElements, all elements will have been read
      _maxNbAccess = nb_elements / _nbElementsToAccessPerIter;
   }

   // access @ref _nbElementsToAccessPerIter elements
   bool _accessElements()
   {
      ++_currentAccess;
      if (_currentAccess != 1)
      {
         Increment<0, false>::run(_iterator_index, _sizeOrder, _pointer_invalid, _nbElementsToAccessPerIter);
      }
      return _currentAccess < _maxNbAccess;
   }

   bool finished() const
   {
      return _currentAccess >= _maxNbAccess;
   }

   // this is the specific view index reordered by <functor>
   const index_type& getIteratorIndex() const
   {
      return _iterator_index;
   }

   // this returns the index in the array
   const index_type getArrayIndex() const
   {
      index_type indexReordered; // start with min offset
      for (size_t n = 0; n < Array::RANK; ++n)
      {
         // currently the index is expressed from fastest to lowest varying speed so transform it back
         indexReordered[n] += _iterator_index[_indexesOrderInv[n]];
      }
      return indexReordered;
   }

   ui32 getVaryingIndex() const
   {
      return _indexesOrder[0];
   }

   const index_type& getVaryingIndexOrder() const
   {
      return _indexesOrder;
   }

   /**
    @brief Force the iterator to stop. Note that at least one more iteration after the break will be executed
    */
   void stop()
   {
      _currentAccess = _maxNbAccess;
   }

protected:
   template <int I, bool B>
   struct Increment
   {
      FORCE_INLINE static void run(StaticVector<ui32, Array::RANK>& index, const StaticVector<ui32, Array::RANK>& size, bool& recomputeIterator,
                                   ui32 nbElements)
      {
         index[I] += nbElements;
         if (index[I] == size[I])
         {
            recomputeIterator = true;
            for (size_t n = 0; n <= I; ++n)
            {
               index[n] = 0;
            }
            Increment<I + 1, (I + 1) == Array::RANK>::run(index, size, recomputeIterator, 1);
            return;
         }
         return;
      }
   };

   template <int I>
   struct Increment<I, true>
   {
      FORCE_INLINE static void run(StaticVector<ui32, Array::RANK>&, const StaticVector<ui32, Array::RANK>&, bool&, ui32)
      {
         return;
      }
   };

protected:
   bool _pointer_invalid = true;
   ui32 _nbElementsToAccessPerIter; // this defines how many elements will be read during a single iteration
   ui32 _maxNbAccess;               // after this number of @p _accessElements calls, all elements will have been accessed
   ui32 _currentAccess = 0;         // so far the current number of @p _accessElements calls
   index_type _iterator_index;      // the current index
   index_type _shape;               // shape of the mapped array
   index_type _sizeOrder;           // the size, ordered by <_indexesOrder>
   index_type _indexesOrder;        // the order of the traversal
   index_type _indexesOrderInv;     // the order of the traversal
};

DECLARE_NAMESPACE_NLL_END
