#pragma once

DECLARE_NAMESPACE_NLL

template <class T, size_t N, class IndexMapper, class Allocator, class PointerType>
class Memory_contiguous;

namespace details
{
template <class Memory>
class ConstMemoryProcessor_contiguous_byMemoryLocality;

template <class Memory>
class MemoryProcessor_contiguous_byMemoryLocality;

/**
 @brief defines if 
 */
template <class Memory>
class MemoryMoveable
{
public:
   // general case: yes we can!
   static bool can_move(typename Memory::value_type*, size_t)
   {
      return true;
   }
};

template <class T, size_t N, class IndexMapper, class Allocator, class PointerType>
class MemoryMoveable<Memory_contiguous<T, N, IndexMapper, Allocator, PointerType>>
{
public:
   using Memory = Memory_contiguous<T, N, IndexMapper, Allocator, PointerType>;

   // Memory_contiguous is known for "stack based memory", so check the allocator
   static bool _can_move(std::true_type UNUSED(base_not_moveable), typename Memory::value_type* ptr, size_t size)
   {
      return Allocator::can_move(ptr, size);
   }

   static bool _can_move(std::false_type UNUSED(moveable), typename Memory::value_type*, size_t)
   {
      return true;
   }

public:
   static bool can_move(typename Memory::value_type* ptr, size_t size)
   {
      return _can_move(std::is_base_of<memory_not_moveable, typename Memory::allocator_type>(), ptr, size);
   }
};
}

/**
@brief Returns true if an array is based on a single slice of contiguous memory
@note this doesn't mean there is not gap between dimensions (e.g., we have a sub-array)
*/
template <class Memory>
struct IsMemoryLayoutContiguous
{
   static const bool value = std::is_base_of<memory_layout_contiguous, Memory>::value;
};

/**
@brief Returns true if an array is based on a single slice or multiple slices of contiguous memory
@note this doesn't mean there is not gap between dimensions (e.g., we have a sub-array)
*/
template <class Memory>
struct IsMemoryLayoutLinear
{
   static const bool value = std::is_base_of<memory_layout_linear, Memory>::value;
};

namespace details
{
template <class Memory>
struct IsMemoryFullyContiguous
{
   static bool value(const Memory& memory, std::integral_constant<bool, true> UNUSED(isContiguous))
   {
      auto stride = memory.getIndexMapper()._getPhysicalStrides();

      std::array<std::pair<ui32, size_t>, Memory::RANK> stride_index;
      for (size_t n = 0; n < Memory::RANK; ++n)
      {
         stride_index[n] = std::make_pair(stride[n], n);
      }

      // sort by increasing stride. Stride must start at 1 and multiplied by the corresponding shape's
      // index

      std::sort(stride_index.begin(), stride_index.end());

      size_t first_index = 0;
      if (stride_index[0].first == 0)
      {
         // discard the <0> so we can share the implementation with the slice based memory
         ++first_index;
      }

      if (stride_index[first_index].first != 1)
      {
         return false;
      }

      ui32 current = 1;
      for (size_t n = first_index; n < Memory::RANK; ++n)
      {
         const auto dim = stride_index[n].second;
         // this test is not perfect when we have a dimension == 1 as we can't know which is
         // the true fastest dimension, so the result of the sort may not be correct, so discard
         // if memory.shape()[dim] != 1
         if (memory.shape()[dim] != 1 && stride_index[n].first != current)
         {
            return false;
         }
         current *= memory.shape()[stride_index[n].second];
      }
      return true;
   }

   static bool value(const Memory& UNUSED(memory), std::integral_constant<bool, false> UNUSED(isContiguous))
   {
      return false;
   }
};
}

/**
@brief Returns true if the array is fully contiguous, meaning that the array occupies a single block of contiguous memory
with no gap between elements (i.e., can't generally be a sub-array)
*/
template <class memory_type>
bool is_memory_fully_contiguous(const memory_type& a1)
{
   if (!IsMemoryLayoutContiguous<memory_type>::value)
   {
      return false;
   }

   // test that the last element of one dimension + 1 equals the first element of the next dimension
   return details::IsMemoryFullyContiguous<memory_type>::value(a1, std::integral_constant<bool, IsMemoryLayoutContiguous<memory_type>::value>());
}

/**
 @brief Extend the allocator to support GPU
 */
template <class Allocator>
struct allocator_traits_gpu_extended : public std::allocator_traits<Allocator>
{
public:
   using value_type     = typename Allocator::value_type;
   using pointer_type   = typename Allocator::pointer;
   using allocator_type = Allocator;

   static void construct_n(size_t linear_size, allocator_type alloc, pointer_type p, value_type default_value)
   {
      _construct_n(linear_size, alloc, p, default_value, is_allocator_gpu<Allocator>());
   }

   static void destroy_n(size_t linear_size, allocator_type alloc, pointer_type p)
   {
      _destroy_n(linear_size, alloc, p, is_allocator_gpu<Allocator>());
   }

private:
   static void _construct_n(size_t linear_size, allocator_type alloc, pointer_type p, value_type default_value, std::false_type)
   {
      // default to std::allocator_traits called for each value
      for (size_t nn = 0; nn < linear_size; ++nn)
      {
         std::allocator_traits<Allocator>::construct(alloc, p + nn, default_value);
      }
   }

   static void _construct_n(size_t linear_size, allocator_type alloc, pointer_type p, value_type default_value, std::true_type)
   {
      // if batch available, call it
      alloc.construct_n(linear_size, p, default_value);
   }

   static void _destroy_n(size_t linear_size, allocator_type alloc, pointer_type p, std::false_type)
   {
      // default to std::allocator_traits called for each value
      for (size_t nn = 0; nn < linear_size; ++nn)
      {
         std::allocator_traits<Allocator>::destroy(alloc, p + nn);
      }
   }

   static void _destroy_n(size_t linear_size, allocator_type alloc, pointer_type p, std::true_type)
   {
      // if batch available, call it
      alloc.destroy_n(p, linear_size);
   }
};

/**
@brief Memory composed of a single memory block

Value based semantics, except when using sub-memory blocks which keeps a reference of the memory.

A sub-block must NOT be accessed when the original memory has been destroyed: dangling pointer!
*/
template <class T, size_t N, class IndexMapper = IndexMapper_contiguous<N>, class Allocator = std::allocator<T>, class PointerType = T*>
class Memory_contiguous : public memory_layout_contiguous
{
   template <class T2, size_t N2, class IndexMapper2, class Allocator2, class PointerType2>
   friend class Memory_contiguous;

public:
   using index_type         = StaticVector<ui32, N>;
   using allocator_type     = Allocator;
   using allocator_traits   = allocator_traits_gpu_extended<allocator_type>;
   using index_mapper       = IndexMapper;
   using pointer_type       = PointerType;
   using const_pointer_type = typename array_add_const<pointer_type>::type;
   using value_type         = T;
   using Memory             = Memory_contiguous<T, N, IndexMapper, Allocator, PointerType>;

   static const size_t RANK = N;

public:
   /**
    @brief Rebind type & dimension
    */
   template <class T2, size_t N2>
   struct rebind_type_dim
   {
      using unconst_type = typename std::remove_const<T2>::type;

      // do NOT use the const in the allocator: this is underfined and won't compile for GCC/Clang
      using other = Memory_contiguous<T2, N2, typename IndexMapper::template rebind<N2>::other, typename Allocator::template rebind<unconst_type>::other,
                                      typename RebindPointer<PointerType, T2>::type>;
   };

   /**
   @brief Rebind the memory with another type
   */
   template <class T2>
   struct rebind
   {
      using other = typename rebind_type_dim<T2, N>::other;
   };

   /**
   @brief Rebind with a different dimension
   */
   template <size_t N2>
   struct rebind_dim
   {
      using other = typename rebind_type_dim<T, N2>::other;
   };

   using ConstMemory = typename Memory::template rebind<const T>::other;

   template <class TT>
   class diterator_t : public std::iterator<std::random_access_iterator_tag, TT>
   {
      friend class diterator_t<const TT>;
      typedef std::iterator<std::random_access_iterator_tag, TT> Base;

      typedef typename Base::difference_type difference_type;
      typedef typename Base::reference reference;
      typedef typename Base::pointer pointer;

   public:
      diterator_t() : _p(nullptr)
      {
      }

      diterator_t(TT* p, size_t stride) : _p(p), _stride(stride)
      {
      }

      diterator_t(const diterator_t<typename std::remove_const<TT>::type>& other) : diterator_t(other._p, other._stride)
      {
      }

      diterator_t& operator++()
      {
         _p += _stride;
         return *this;
      }

      diterator_t& add(int step = 1)
      {
         _p += _stride * step;
         return *this;
      }

      bool operator==(const diterator_t& rhs) const
      {
         NLL_FAST_ASSERT(_stride == rhs._stride, "non matching <D> iterator");
         return rhs._p == _p;
      }

      bool operator!=(const diterator_t& rhs) const
      {
         NLL_FAST_ASSERT(_stride == rhs._stride, "non matching <D> iterator");
         return rhs._p != _p;
      }

      difference_type operator-(const diterator_t& rhs) const
      {
         NLL_FAST_ASSERT(_stride == rhs._stride, "non matching <D> iterator");
         return (_p - rhs._p) / _stride;
      }

      reference operator*()
      {
         return *_p;
      }

      pointer current_pointer() const
      {
         return _p;
      }

   private:
      pointer _p;
      size_t _stride;
   };

   using diterator       = diterator_t<T>;
   using const_diterator = diterator_t<const T>;

   Memory_contiguous(const allocator_type& allocator = allocator_type()) : _allocator(allocator)
   {
   }

   /// New memory block
   Memory_contiguous(const index_type& shape, value_type default_value = value_type(), const allocator_type& allocator = allocator_type())
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _allocateSlices(default_value, _linearSize());
   }

   /**
   @param slices pre-existing slices. if @p slicesAllocated, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, @p allocator will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_contiguous(const index_type& shape, pointer_type data, const allocator_type& allocator = allocator_type(), bool dataAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _data          = data;
      _dataAllocated = dataAllocated;
   }

   /**
   @param data pre-existing memory elements. if @p dataAllocated, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param dataAllocated if true, @p allocator will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_contiguous(const index_type& shape, pointer_type data, const index_type& physicalStrides, const allocator_type& allocator = allocator_type(),
                     bool dataAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(physicalStrides);
      _data          = data;
      _dataAllocated = dataAllocated;
   }

   template <size_t dimension>
   struct slice_type
   {
      using type = typename rebind_type_dim<T, N - 1>::other;
   };

   /**
   @brief Slice the memory such that we keep only the slice along dimension @p dimension passing through @p index

   Create a reference of this object, so do NOT destroy the memory while using the sliced mempory
   */
   template <size_t dimension>
   typename slice_type<dimension>::type slice(const index_type& index) const
   {
      // we start at the beginning of the slice
      index_type index_slice;
      index_slice[dimension] = index[dimension];
      pointer_type ptr       = pointer_type(this->at(index_slice));

      using Other = typename slice_type<dimension>::type;
      Other memory;
      memory._indexMapper   = _indexMapper.slice<dimension>(index_slice);
      memory._data          = pointer_type(ptr);
      memory._dataAllocated = false; // this is a "reference"
      memory._allocator     = getAllocator();

      size_t current_dim = 0;
      for (size_t n = 0; n < N; ++n)
      {
         if (n != dimension)
         {
            memory._shape[current_dim++] = _shape[n];
         }
      }
      return memory;
   }

   /**
   @brief reference an existing sub-memory block
   @param strides the stride (spacing between data elements) to be used to access the @p ref. (i.e., this doesn't depend on the
   stride of @p ref itself
   @param shape the size of the actual memory
   @param min_index the index in @p ref to be used as the first data element
   */
   Memory_contiguous(Memory_contiguous& ref, const index_type& min_index, const index_type& shape, const index_type& strides) : _shape(shape)
   {
      _data          = ref._data;
      _dataAllocated = false; // we create a reference on existing slices
      _indexMapper   = ref.getIndexMapper().submap(min_index, shape, strides);

      if (ref._sharedView)
      {
         // there is already a reference, keep it
         _sharedView = ref._sharedView;
      }
      else
      {
         // new reference
         _sharedView = &ref;
      }
   }

   ConstMemory asConst() const
   {
      return ConstMemory(shape(), _data, _indexMapper._getPhysicalStrides());
   }

   diterator beginDim(ui32 dim, const index_type& indexN)
   {
      auto p = this->at(indexN);
      return diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   const_diterator beginDim(ui32 dim, const index_type& indexN) const
   {
      auto p = this->at(indexN);
      return const_diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   diterator endDim(ui32 dim, const index_type& indexN)
   {
      index_type index_cpy = indexN;
      index_cpy[dim]       = this->_shape[dim];

      auto p = this->at(index_cpy);
      return diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   const_diterator endDim(ui32 dim, const index_type& indexN) const
   {
      index_type index_cpy = indexN;
      index_cpy[dim]       = this->_shape[dim];

      auto p = this->at(index_cpy);
      return const_diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

private:
   ui32 _linearSize() const
   {
      ui32 size = 1;
      for (size_t n = 0; n < N; ++n)
      {
         size *= _shape[n];
      }
      return size;
   }

   void _allocateSlices(value_type default_value, ui32 linear_size)
   {
      auto p = allocator_traits::allocate(_allocator, linear_size);
      allocator_traits::construct_n(linear_size, _allocator, p, default_value);
      _data = pointer_type(p);
   }

   void _deallocateSlices()
   {
      if (_dataAllocated)
      {
         const auto linear_size = _linearSize();

         // handle the const pointer_type case for const arrays
         using unconst_value = typename std::remove_cv<T>::type;

         auto ptr = typename allocator_type::pointer(_data);
         allocator_traits::destroy_n(linear_size, _allocator, ptr);
         allocator_traits::deallocate(_allocator, ptr, linear_size);
      }

      _data       = pointer_type(nullptr);
      _sharedView = nullptr;
   }

   void _moveCopy(Memory_contiguous&& other)
   {
      if (this != &other)
      {
         _indexMapper = other._indexMapper;
         _shape       = other._shape;
         _allocator   = other._allocator;

         _dataAllocated       = other._dataAllocated;
         other._dataAllocated = false;

         _data       = other._data;
         other._data = pointer_type(nullptr);

         _sharedView = other._sharedView;
      }
   }

   template <class Allocator2, class PointerType2>
   void _deepCopy_notallocator(const Memory_contiguous<value_type, N, index_mapper, Allocator2, PointerType2>& other)
   {
      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _shape = other._shape;
      _indexMapper.init(0, _shape);
      _dataAllocated = true;

      // now deep copy...
      const ui32 this_linearSize  = _linearSize();
      const ui32 other_linearSize = other._sharedView ? other._sharedView->_linearSize() : other._linearSize();
      if (other_linearSize == 0)
      {
         // empty other. Done!
         return;
      }

      using unconst_pointer = typename array_remove_const<pointer_type>::type;

      _allocateSlices(value_type(), this_linearSize);
      // TODO need to improve the detection mechanism! This is currently too restrictive.W e want to allow multidimentional
      // array with same stride (i.e., no gap between dimensions)
      if (this_linearSize == other_linearSize && is_memory_fully_contiguous(other) &&
          same_data_ordering_memory(other, *this)) // if we have a non stride (1,...,1) stride, use iterator
      {
         // this means the deep copy is the FULL buffer
         static_assert(std::is_standard_layout<value_type>::value, "must have standard layout!");
         const auto src = other._data;
         const auto dst = _data;
         NLL_FAST_ASSERT(!std::is_const<value_type>::value,
                         "type is CONST!"); // _deepCopy can be compiled with a const array, but it should actually never run it...
         details::copy_naive(unconst_pointer(dst), 1, src, 1, this_linearSize);
      }
      else
      {
         // we have a subarray, potentially with stride so we need to use a processor
         using const_pointer_type = typename array_add_const<PointerType2>::type;
         auto op_cpy              = [&](pointer_type y_pointer, ui32 y_stride, const_pointer_type x_pointer, ui32 x_stride, ui32 nb_elements) {
            // @TODO add the BLAS copy
            NLL_FAST_ASSERT(!std::is_const<value_type>::value, "type is CONST!");
            details::copy_naive(unconst_pointer(y_pointer), y_stride, x_pointer, x_stride, nb_elements);
         };
         iterate_memory_constmemory(*this, other, op_cpy);
      }
   }

   template <class PointerType2>
   void _deepCopy(const Memory_contiguous<value_type, N, index_mapper, allocator_type, PointerType2>& other)
   {
      _deepCopy_notallocator(other);
      _allocator = other._allocator;
   }

public:
   Memory_contiguous(const Memory_contiguous& other)
   {
      _deepCopy(other);
   }

   template <class PointerType2>
   Memory_contiguous(const Memory_contiguous<value_type, N, index_mapper, allocator_type, PointerType2>& other)
   {
      _deepCopy(other);
   }

   template <class Allocator2, class PointerType2>
   Memory_contiguous(const Memory_contiguous<value_type, N, index_mapper, Allocator2, PointerType2>& other)
   {
      _deepCopy_notallocator(other);
   }

   Memory_contiguous& operator=(const Memory_contiguous& other)
   {
      _deepCopy(other);
      return *this;
   }

   template <class PointerType2>
   Memory_contiguous& operator=(const Memory_contiguous<value_type, N, index_mapper, allocator_type, PointerType2>& other)
   {
      _deepCopy(other);
      return *this;
   }

   template <class Allocator2, class PointerType2>
   Memory_contiguous& operator=(const Memory_contiguous<value_type, N, index_mapper, Allocator2, PointerType2>& other)
   {
      _deepCopy_notallocator(other);
      return *this;
   }

   Memory_contiguous(Memory_contiguous&& other)
   {
      // make sure we don't have non-moveable memory
      if (details::MemoryMoveable<Memory_contiguous>::can_move(other._data, other._linearSize()))
      {
         _moveCopy(std::forward<Memory_contiguous>(other));
      }
      else
      {
         _deepCopy(other);
      }
   }

   Memory_contiguous& operator=(Memory_contiguous&& other)
   {
      // make sure we don't have non-moveable memory
      if (details::MemoryMoveable<Memory_contiguous>::can_move(other._data, other._linearSize()))
      {
         _moveCopy(std::forward<Memory_contiguous>(other));
      }
      else
      {
         _deepCopy(other);
      }
      return *this;
   }

   ~Memory_contiguous()
   {
      _deallocateSlices();
   }

   const pointer_type at(const index_type& index) const
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   pointer_type at(const index_type& index)
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   const index_type& shape() const
   {
      return _shape;
   }

   const IndexMapper& getIndexMapper() const
   {
      return _indexMapper;
   }

   IndexMapper& getIndexMapper()
   {
      return _indexMapper;
   }

   const allocator_type& getAllocator() const
   {
      return _allocator;
   }

   size_t size() const
   {
      size_t s = 1;
      for (int n = 0; n < N; ++n)
      {
         s *= _shape[n];
      }
      return s;
   }

private:
   // arrange py decreasing size order to help with the structure packing
   IndexMapper _indexMapper;
   pointer_type _data             = nullptr; /// BEWARE *_data may not be equal to at(index_type()) as there may be an offset in the index mapper
   Memory_contiguous* _sharedView = nullptr; /// the original array
   index_type _shape;
   allocator_type _allocator;
   bool _dataAllocated = true;
};

template <class T, size_t N, class Allocator = std::allocator<T>>
using Memory_contiguous_row_major = Memory_contiguous<T, N, IndexMapper_contiguous_row_major<N>, Allocator>;

template <class T, size_t N, class Allocator = std::allocator<T>>
using Memory_contiguous_column_major = Memory_contiguous<T, N, IndexMapper_contiguous_column_major<N>, Allocator>;

#ifdef WITH_CUDA
/**
 @tparam ___IndexMapper make no mistake: cuda based memory ONLY support column-major ordering! However, we can use
         a different mapper for the same column-major ordering (e.g., matrix)
*/
template <class T, size_t N, class Allocator = AllocatorCuda<T>, class ___IndexMapper = IndexMapper_contiguous_column_major<N>>
using Memory_cuda_contiguous_column_major = Memory_contiguous<T, N, ___IndexMapper, Allocator, cuda_ptr<T>>;
#endif

DECLARE_NAMESPACE_NLL_END
