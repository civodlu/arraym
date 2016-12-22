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

   template <class FunctorGetDimensionOrder>
   ArrayChunking_contiguous_base(Array& array, const FunctorGetDimensionOrder& functor) : _array(array)
   {
      _indexesOrder = functor(array);
      for (ui32 n = 0; n < _indexesOrder.size(); ++n)
      {
         _indexesOrderInv[_indexesOrder[n]] = n;
         _sizeOrder[n] = array.shape()[_indexesOrder[n]];
      }
      _maxAccessElements = array.shape()[_indexesOrder[0]];
   }

   // <nbElements> minimum = 1
   //              maximum = max number of elements contiguous in the fastest varying dimension
   bool _accessElements(ui32 nbElements)
   {
      NLL_FAST_ASSERT(nbElements == 1 || nbElements == _maxAccessElements, "TODO handle different nbElements!");
      const bool hasMoreElements = Increment<0, false>::run(_iterator_index, _sizeOrder, _pointer_invalid, nbElements);
      return hasMoreElements;
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

protected:
   template <int I, bool B>
   struct Increment
   {
      FORCE_INLINE static bool run(StaticVector<ui32, Array::RANK>& index, const StaticVector<ui32, Array::RANK>& size, bool& recomputeIterator,
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
            return Increment<I + 1, (I + 1) == Array::RANK>::run(index, size, recomputeIterator, 1);
         }
         return true;
      }
   };

   template <int I>
   struct Increment<I, true>
   {
      FORCE_INLINE static bool run(StaticVector<ui32, Array::RANK>&, const StaticVector<ui32, Array::RANK>&, bool&, ui32)
      {
         return false;
      }
   };

protected:
   bool       _pointer_invalid = true;
   ui32       _maxAccessElements; // maximum number of steps in the fastest varying dimension possible without increasing the other indexes
   index_type _iterator_index;  // the current index
   Array&     _array;
   index_type _sizeOrder;       // the size, ordered by <_indexesOrder>
   index_type _indexesOrder;    // the order of the traversal
   index_type _indexesOrderInv; // the order of the traversal
};

DECLARE_NAMESPACE_NLL_END