#pragma once

DECLARE_NAMESPACE_NLL

/**
 @file

 This file defines "processors", which allow to iterate over an array or multiple arrays in the most efficient manner considering
 the memory locality of the arrays
 */

template <class Array>
class ConstArrayProcessor_contiguous_byMemoryLocality;

template <class Array>
class ConstArrayProcessor_contiguous_base;

template <class Array>
class ConstMemoryProcessor_contiguous_byMemoryLocality;

template <class Array>
class ConstMemoryProcessor_contiguous_base;

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
class ArrayProcessor_contiguous_base : public ArrayChunking_contiguous_base<Array>
{
   using Base = ArrayChunking_contiguous_base<Array>;
   using diterator = typename Array::diterator;
   static_assert(std::is_base_of<memory_layout_linear, typename Array::Memory>::value, "must be a linear index mapper!");

public:
   using index_type   = typename Array::index_type;
   using pointer_type = typename Array::pointer_type;

   template <class FunctorGetDimensionOrder>
   ArrayProcessor_contiguous_base(Array& array, const FunctorGetDimensionOrder& functor, ui32 nbElementsToAccessPerIter) : Base(array.shape(), functor(array), nbElementsToAccessPerIter), _array(array)
   {}

   // TODO deprecate
   bool accessSingleElement(pointer_type& ptrToValue)
   {
      return _accessElements(ptrToValue);
   }

   bool _accessElements(pointer_type& ptrToValue)
   {
      if ( this->_pointer_invalid )
      {
         _iterator = this->_array.beginDim( this->_indexesOrder[ 0 ], this->getArrayIndex() );
         this->_pointer_invalid = false;
      }
      else
      {
         _iterator.add(this->_nbElementsToAccessPerIter);
      }
      ptrToValue = pointer_type(_iterator.current_pointer());
      return this->Base::_accessElements();
   }

protected:
   
public:
   // TODO fix the access as protected
   Array& _array;
   
protected:
   diterator _iterator;
   
};

/**
 @brief Const array processor

 This is just an adaptor of @ref ArrayProcessor_contiguous_base
 */
template <class Array>
class ConstArrayProcessor_contiguous_base
{
public:
   using index_type   = typename Array::index_type;
   using pointer_type = typename Array::pointer_type;
   using const_pointer_type = typename Array::const_pointer_type;
   using value_type   = typename Array::value_type;

   template <class FunctorGetDimensionOrder>
   ConstArrayProcessor_contiguous_base(const Array& array, const FunctorGetDimensionOrder& functor, ui32 nbElementsToAccessPerIter) : _processor(const_cast<Array&>(array), functor, nbElementsToAccessPerIter)
   {
   }

   bool accessSingleElement( const_pointer_type& ptrToValue )
   {
      return _processor.accessSingleElement( (pointer_type&)( ptrToValue ) );
   }

   // this is the specific view index reordered by <functor>
   const index_type& getIteratorIndex() const
   {
      return _processor.getIteratorIndex();
   }

   const index_type getArrayIndex() const
   {
      return _processor.getArrayIndex();
   }

   ui32 getVaryingIndex() const
   {
      return _processor.getVaryingIndex();
   }

   const index_type& getVaryingIndexOrder() const
   {
      return _processor.getVaryingIndexOrder();
   }

protected:
   bool _accessElements(const_pointer_type& ptrToValue)
   {
      return _processor._accessElements(const_cast<pointer_type&>(ptrToValue));
   }

protected:
   ArrayProcessor_contiguous_base<Array> _processor;
};

template <class Memory>
StaticVector<ui32, Memory::RANK> getFastestVaryingIndexesMemory(const Memory& memory)
{
   static const size_t N = Memory::RANK;
   using index_type      = typename Memory::index_type;

   static_assert(std::is_base_of<memory_layout_linear, Memory>::value, "must be a linear index mapper!");

   index_type fastestVaryingIndexes;

   // first, we want to iterate from the fastest->lowest varying index to avoid as much cache misses as possible
   // EXCEPT is stride is 0, which is a special case (different slices in memory, so this is actually the WORST dimension to iterate on)
   auto strides = memory.getIndexMapper()._getPhysicalStrides();
   for (auto& v : strides)
   {
      if (v == 0)
      {
         v = std::numeric_limits<typename index_type::value_type>::max();
      }
   }

   std::array<std::pair<ui32, ui32>, N> stridesIndex;
   for (ui32 n = 0; n < N; ++n)
   {
      stridesIndex[n] = std::make_pair(strides[n], n);
   }
   std::sort(stridesIndex.begin(), stridesIndex.end());

   for (ui32 n = 0; n < N; ++n)
   {
      fastestVaryingIndexes[n] = stridesIndex[n].second;
   }

   return fastestVaryingIndexes;
}

template <class T, size_t N, class ConfigT>
StaticVector<ui32, N> getFastestVaryingIndexes(const Array<T, N, ConfigT>& array)
{
   return getFastestVaryingIndexesMemory(array.getMemory());
}
}

/**
 @brief iterate an array by maximizing memory locality. This should be the preferred iterator
 */
template <class Array>
class ArrayProcessor_contiguous_byMemoryLocality : public details::ArrayProcessor_contiguous_base<Array>
{
public:
   using base         = details::ArrayProcessor_contiguous_base<Array>;
   using pointer_type = typename base::pointer_type;

   ArrayProcessor_contiguous_byMemoryLocality(Array& array, ui32 nbElementsToAccessPerIter)
      : base(array, &details::getFastestVaryingIndexes<typename Array::value_type, Array::RANK, typename Array::Config>, nbElementsToAccessPerIter)
   {
   }

   ui32 getNbElementsPerAccess() const
   {
      return this->_nbElementsToAccessPerIter;
   }

   ui32 stride() const
   {
      return this->_array.getMemory().getIndexMapper()._getPhysicalStrides()[this->getVaryingIndex()];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, @p ptrToValue if accessed in a contiguous fashion must account for the stride in the direction of access using stride()
   */
   bool accessMaxElements(pointer_type& ptrToValue)
   {
      return this->_accessElements(ptrToValue);
   }
};

/**
@brief iterate an array's memory by maximizing memory locality. This should be the preferred iterator
@note memory behaves like an array so we can reuse the array's processor
*/
template <class Memory>
class MemoryProcessor_contiguous_byMemoryLocality : public details::ArrayProcessor_contiguous_base<Memory>
{
public:
   using base         = details::ArrayProcessor_contiguous_base<Memory>;
   using pointer_type = typename base::pointer_type;

   MemoryProcessor_contiguous_byMemoryLocality(Memory& array, ui32 nbElementsToAccessPerIter) : base(array, &details::getFastestVaryingIndexesMemory<Memory>, nbElementsToAccessPerIter)
   {
   }

   ui32 getNbElementsPerAccess() const
   {
      return this->_nbElementsToAccessPerIter;
   }

   ui32 stride() const
   {
      return this->_array.getIndexMapper()._getPhysicalStrides()[this->getVaryingIndex()];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, @p ptrToValue if accessed in a contiguous fashion must account for the stride in the direction of access using <stride()>
   */
   bool accessMaxElements(pointer_type& ptrToValue)
   {
      return this->_accessElements(ptrToValue);
   }
};

/**
@brief iterate an array's memory by maximizing memory locality. This should be the preferred iterator

This is a const version of @ref ArrayProcessor_contiguous_byMemoryLocality
@note memory behaves like an array so we can reuse the array's processor
*/
template <class Memory>
class ConstMemoryProcessor_contiguous_byMemoryLocality : public details::ConstArrayProcessor_contiguous_base<Memory>
{
public:
   using base         = details::ConstArrayProcessor_contiguous_base<Memory>;
   using pointer_type = typename Memory::pointer_type;
   using const_pointer_type = typename Memory::const_pointer_type;
   using value_type   = typename Memory::value_type;
   
   ConstMemoryProcessor_contiguous_byMemoryLocality(const Memory& array, ui32 nbElementsToAccessPerIter) : base(array, &details::getFastestVaryingIndexesMemory<Memory>, nbElementsToAccessPerIter)
   {
   }

   ui32 getNbElementsPerAccess() const
   {
      return this->_processor._nbElementsToAccessPerIter;
   }

   ui32 stride() const
   {
      return this->_processor._array.getIndexMapper()._getPhysicalStrides()[this->getVaryingIndex()];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, @p ptrToValue if accessed in a contiguous fashion must account for the stride in the direction of access using stride()
   */
   bool accessMaxElements(const_pointer_type& ptrToValue)
   {
      return this->_accessElements(ptrToValue);
   }
};

/**
@brief iterate an array by maximizing memory locality. This should be the preferred iterator

This is a const version of @ref ArrayProcessor_contiguous_byMemoryLocality
*/
template <class Array>
class ConstArrayProcessor_contiguous_byMemoryLocality : public details::ConstArrayProcessor_contiguous_base<Array>
{
public:
   using base         = details::ConstArrayProcessor_contiguous_base<Array>;
   using pointer_type = typename Array::pointer_type;
   using const_pointer_type = typename Array::const_pointer_type;
   using value_type   = typename Array::value_type;

   ConstArrayProcessor_contiguous_byMemoryLocality(const Array& array, ui32 nbElementsToAccessPerIter)
      : base(array, &details::getFastestVaryingIndexes<typename Array::value_type, Array::RANK, typename Array::Config>, nbElementsToAccessPerIter)
   {
   }

   ui32 getNbElementsPerAccess() const
   {
      return this->_processor._nbElementsToAccessPerIter;
   }

   ui32 stride() const
   {
      return this->_processor._array.getMemory().getIndexMapper()._getPhysicalStrides()[this->getVaryingIndex()];
   }

   /**
   @return true if more elements are to be processed

   This is defined only for memory locality as this is the only method guarantying contiguous memory access

   IMPORTANT, @p ptrToValue if accessed in a contiguous fashion must account for the stride in the direction of access using stride()
   */
   bool accessMaxElements(const_pointer_type& ptrToValue)
   {
      return this->_accessElements(ptrToValue);
   }
};

/**
@brief iterate by dimension, in (x, y, z...) order
*/
template <class Array>
class ArrayProcessor_contiguous_byDimension : public details::ArrayProcessor_contiguous_base<Array>
{
public:
   using base       = details::ArrayProcessor_contiguous_base<Array>;
   using index_type = typename base::index_type;
   static index_type getIndexes(const Array&)
   {
      index_type indexes;
      for (ui32 n = 0; n < Array::RANK; ++n)
      {
         indexes[n] = n;
      }
      return indexes;
   }

public:
   ArrayProcessor_contiguous_byDimension(Array& array, ui32 nbElementsToAccessPerIter) : details::ArrayProcessor_contiguous_base<Array>(array, &getIndexes, nbElementsToAccessPerIter)
   {
   }
};

/**
@brief iterate by dimension, in (x, y, z...) order
*/
template <class Array>
class ConstArrayProcessor_contiguous_byDimension : public details::ConstArrayProcessor_contiguous_base<Array>
{
public:
   using base         = details::ConstArrayProcessor_contiguous_base<Array>;
   using pointer_type = typename base::pointer_type;
   using index_type   = typename base::index_type;

   static index_type getIndexes(const Array&)
   {
      index_type indexes;
      for (ui32 n = 0; n < Array::RANK; ++n)
      {
         indexes[n] = n;
      }
      return indexes;
   }

   ConstArrayProcessor_contiguous_byDimension(const Array& array, ui32 nbElementsToAccessPerIter) : base(array, &getIndexes, nbElementsToAccessPerIter)
   {
   }
};

template <class T>
struct CompileError
{
   static_assert(std::is_same<T, char>::value, "Compile error purpose for T!");
};
namespace impl
{
   template <class Memory1, class Memory2, class Op, typename = typename std::enable_if<IsMemoryLayoutLinear<Memory1>::value>::type>
   void _iterate_memory_constmemory_same_ordering(Memory1& a1, const Memory2& a2, const Op& op)
   {
      using pointer_T = typename Memory1::pointer_type;
      using pointer_T2 = typename Memory2::pointer_type;
      using pointer_const_T2 = typename array_add_const<pointer_T2>::type;
      static const size_t N = Memory1::RANK;

      static_assert(is_callable_with<Op, pointer_T, ui32, pointer_const_T2, ui32, ui32>::value, "Op is not callable!");
      ensure(Memory1::RANK == Memory2::RANK, "must have the same rank!");
      ensure(a1.shape() == a2.shape(), "must have the same shape!");
      ensure(same_data_ordering_memory(a1, a2), "data must have a similar ordering!");

      // we MUST use processors: data may not be contiguous or with stride...
      ConstMemoryProcessor_contiguous_byMemoryLocality<Memory2> processor_a2(a2, 0);
      MemoryProcessor_contiguous_byMemoryLocality<Memory1> processor_a1(a1, 0);

      bool hasMoreElements = true;
      while (hasMoreElements)
      {
         pointer_T ptr_a1 = pointer_T(nullptr);
         pointer_const_T2 ptr_a2 = pointer_const_T2(nullptr);
         static_assert(std::is_same<pointer_const_T2, typename ConstMemoryProcessor_contiguous_byMemoryLocality<Memory2>::const_pointer_type>::value, "must be the same!");

         hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
         hasMoreElements = processor_a2.accessMaxElements(ptr_a2);
         NLL_FAST_ASSERT(processor_a1.getNbElementsPerAccess() == processor_a2.getNbElementsPerAccess(), "memory line must have the same size");

         op(ptr_a1, processor_a1.stride(), ptr_a2, processor_a2.stride(), processor_a1.getNbElementsPerAccess());
      }
   }

   template <class Memory1, class Memory2, class Op, typename = typename std::enable_if<IsMemoryLayoutLinear<Memory1>::value>::type>
   void _iterate_memory_constmemory_different_ordering(Memory1& a1, const Memory2& a2, const Op& op)
   {
      using pointer_T = typename Memory1::pointer_type;
      using pointer_T2 = typename Memory2::pointer_type;
      using pointer_const_T2 = typename array_add_const<pointer_T2>::type;
      static const size_t N = Memory1::RANK;

      static_assert( is_callable_with<Op, pointer_T, ui32, pointer_const_T2, ui32, ui32>::value, "Op is not callable!" );

      ensure(Memory1::RANK == Memory2::RANK, "must have the same rank!");
      ensure(a1.shape() == a2.shape(), "must have the same shape!");
      ensure(!same_data_ordering_memory(a1, a2), "data must have a similar ordering!");

      // we MUST use processors: data may not be contiguous or with stride...
      // additionally the order of dimensions are different, so map the a2 order
      ConstMemoryProcessor_contiguous_byMemoryLocality<Memory2> processor_a2(a2, 1);
      auto functor_order = [&](const Memory1&)
      {
         return processor_a2.getVaryingIndexOrder();
      };
      
      details::ArrayProcessor_contiguous_base<Memory1> processor_a1(a1, functor_order, 1);
      
      bool hasMoreElements = true;
      while (hasMoreElements)
      {
         pointer_T ptr_a1 = pointer_T(nullptr);
         pointer_const_T2 ptr_a2 = pointer_const_T2(nullptr);
         hasMoreElements = processor_a1.accessSingleElement(ptr_a1);
         hasMoreElements = processor_a2.accessSingleElement(ptr_a2);
         op(ptr_a1, 1, ptr_a2, 1, 1); // only single element, so actual stride value is not important, it just can't be 0
      }
   }
}

/**
@brief iterate array & const array jointly
@tparam must be callable using (pointer_type a1_pointer, ui32 a1_stride, const_pointer_type a2_pointer, ui32 a2_stride, ui32 nb_elements)
@note this is only instantiated for linear memory
*/
template <class Memory1, class Memory2, class Op, typename = typename std::enable_if<IsMemoryLayoutLinear<Memory1>::value>::type>
void iterate_memory_constmemory(Memory1& a1, const Memory2& a2, const Op& op)
{
   if (same_data_ordering_memory(a1, a2))
   {
      impl::_iterate_memory_constmemory_same_ordering(a1, a2, op);
   }
   else {
      impl::_iterate_memory_constmemory_different_ordering(a1, a2, op);
   }
}

/**
@brief iterate array & const array jointly
@tparam Op must be callable using (pointer_type a1_pointer, ui32 a1_stride, const_pointer_type a2_pointer, ui32 a2_stride, ui32 nb_elements)
@note this is only instantiated for linear memory
*/
template <class T, class T2, size_t N, class Config, class Config2, class Op,
          typename = typename std::enable_if<IsArrayLayoutLinear<Array<T, N, Config>>::value>::type>
void iterate_array_constarray(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2, Op& op)
{
   iterate_memory_constmemory(a1.getMemory(), a2.getMemory(), op);
}

namespace details
{
   template <class T, class T2, size_t N, class Config, class Config2, class Op>
   void _iterate_array_constarray(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2, Op& op)
   {
      iterate_memory_constmemory(a1.getMemory(), a2.getMemory(), op);
   }
}

/**
@brief iterate array
@tparam must be callable using (pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
@note this is only instantiated for linear memory
*/
template <class T, size_t N, class Config, class Op, typename = typename std::enable_if<IsArrayLayoutLinear<Array<T, N, Config>>::value>::type>
void iterate_array(Array<T, N, Config>& a1, Op& op)
{
   using array_type = Array<T, N, Config>;
   using pointer_type = typename array_type::pointer_type;
   ArrayProcessor_contiguous_byMemoryLocality<array_type> processor_a1(a1, 0);

   static_assert(is_callable_with<Op, pointer_type, ui32, ui32>::value, "Op is not callable with the correct arguments!");

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      pointer_type ptr_a1(nullptr);
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      op(ptr_a1, processor_a1.stride(), processor_a1.getNbElementsPerAccess());
   }
}

/**
@brief iterate array
@tparam must be callable using (const_pointer_type a1_pointer, ui32 a1_stride, ui32 nb_elements)
@note this is only instantiated for linear memory
*/
template <class T, size_t N, class Config, class Op, typename = typename std::enable_if<IsArrayLayoutLinear<Array<T, N, Config>>::value>::type>
void iterate_constarray(const Array<T, N, Config>& a1, Op& op)
{
   using array_type = Array<T, N, Config>;
   using const_pointer_type = typename array_type::const_pointer_type;
   ConstArrayProcessor_contiguous_byMemoryLocality<array_type> processor_a1(a1, 0);

   static_assert(is_callable_with<Op, const_pointer_type, ui32, ui32>::value, "Op is not callable with the correct arguments!");

   bool hasMoreElements = true;
   while (hasMoreElements)
   {
      const_pointer_type ptr_a1(nullptr);
      hasMoreElements = processor_a1.accessMaxElements(ptr_a1);
      op(ptr_a1, processor_a1.stride(), processor_a1.getNbElementsPerAccess());
   }
}
DECLARE_NAMESPACE_NLL_END