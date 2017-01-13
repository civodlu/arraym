#pragma once

#ifdef WITH_CUDA


DECLARE_NAMESPACE_NLL

template <class T, size_t N, class Allocator>
class Memory_gpu_cuda;



template <class T, size_t N, class Allocator = AllocatorCuda<T>>
class Memory_gpu_cuda : public memory_layout_contiguous
{
   template <class T2, size_t N2, class Allocator2>
   friend class Memory_gpu_cuda;

public:
   using index_type = StaticVector<ui32, N>;
   using allocator_type = Allocator;
   using allocator_trait = std::allocator_traits<allocator_type>;
   using index_mapper = IndexMapper_contiguous<N, details::Mapper_stride_column_major<N>>; // CUDA supports only column-major memory style
   using pointer_type = T*;
   using value_type = T;
   using Memory = Memory_gpu_cuda<T, N, Allocator>;

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
      using other = Memory_gpu_cuda<T2, N2, typename Allocator::template rebind<unconst_type>::other>;
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

      /*
      reference operator*()
      {
         return *_p;
      }*/

   private:
      pointer _p;
      size_t _stride;
   };

   using diterator = diterator_t<T>;
   using const_diterator = diterator_t<const T>;

   Memory_gpu_cuda(const allocator_type& allocator = allocator_type()) : _allocator(allocator)
   {
   }

   /// New memory block
   Memory_gpu_cuda(const index_type& shape, T default_value = T(), const allocator_type& allocator = allocator_type()) : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _allocateSlices(default_value, _linearSize());
   }

   /**
   @param slices pre-existing slices. if @p slicesAllocated, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, @p allocator will be used to deallocate the memory. Else the user is responsible for the slice's memory
   
   TODO
   - take advantage of the streams, don't use default that will sync with the others
   - memory allocation is slow, use custom allocator
   
   */
   Memory_gpu_cuda(const index_type& shape, T* data, const allocator_type& allocator = allocator_type(), bool dataAllocated = false)
      : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _data = data;
      _dataAllocated = dataAllocated;
   }

   /**
   @param data pre-existing memory elements. if @p dataAllocated, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param dataAllocated if true, @p allocator will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_gpu_cuda(const index_type& shape, T* data, const index_type& physicalStrides, const allocator_type& allocator = allocator_type(),
      bool dataAllocated = false)
      : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(physicalStrides);
      _data = data;
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
      T* ptr = const_cast<T*>(this->at(index_slice));

      using Other = typename slice_type<dimension>::type;
      Other memory;
      memory._indexMapper = _indexMapper.slice<dimension>(index_slice);
      memory._data = const_cast<T*>(ptr);
      memory._dataAllocated = false; // this is a "reference"
      memory._allocator = getAllocator();

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
   Memory_gpu_cuda(Memory_gpu_cuda& ref, const index_type& min_index, const index_type& shape, const index_type& strides) : _shape(shape)
   {
      _data = ref._data;
      _dataAllocated = false; // we create a reference on existing slices
      _indexMapper = ref.getIndexMapper().submap(min_index, shape, strides);

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
      index_cpy[dim] = this->_shape[dim];

      auto p = this->at(index_cpy);
      return diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   const_diterator endDim(ui32 dim, const index_type& indexN) const
   {
      index_type index_cpy = indexN;
      index_cpy[dim] = this->_shape[dim];

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

   void _allocateSlices(T default_value, ui32 linear_size)
   {
      // TODO handle array of vectors
      _data = allocator_trait::allocate(_allocator, linear_size);
      cuda::kernel_init(_data, default_value, linear_size);
   }

   void _deallocateSlices()
   {
      if (_dataAllocated)
      {
         const auto linear_size = _linearSize();
         for (size_t nn = 0; nn < linear_size; ++nn)
         {
            allocator_trait::destroy(_allocator, _data + nn);
         }
         // handle the const T* case for const arrays
         using unconst_value = typename std::remove_cv<T>::type;
         allocator_trait::deallocate(_allocator, const_cast<unconst_value*>(_data), linear_size);
      }

      _data = nullptr;
      _sharedView = nullptr;
   }

   void _moveCopy(Memory_gpu_cuda&& other)
   {
      if (this != &other)
      {
         _indexMapper = other._indexMapper;
         _shape = other._shape;
         _allocator = other._allocator;

         _dataAllocated = other._dataAllocated;
         other._dataAllocated = false;

         _data = other._data;
         other._data = nullptr;

         _sharedView = other._sharedView;
      }
   }

   void _deepCopy(const Memory_gpu_cuda& other)
   {
      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _shape = other._shape;
      _indexMapper.init(0, _shape);
      _allocator = other._allocator;
      _dataAllocated = true;

      // now deep copy...
      const ui32 this_linearSize = _linearSize();
      const ui32 other_linearSize = other._sharedView ? other._sharedView->_linearSize() : other._linearSize();
      using unconst_type = typename std::remove_const<T>::type;

      _allocateSlices(T(), this_linearSize);
      if (this_linearSize == other_linearSize && is_memory_fully_contiguous(other) && same_data_ordering_memory(other, *this)) // if we have a non stride (1,...,1) stride, use iterator
      {
         // this means the deep copy is the FULL buffer
         const auto size_bytes = sizeof(T) * this_linearSize;
         static_assert(std::is_standard_layout<T>::value, "must have standard layout!");
         const auto src = other._data;
         const auto dst = _data;
         NLL_FAST_ASSERT(!std::is_const<T>::value, "type is CONST!");  // _deepCopy can be compiled with a const array, but it should actually never run it...
         memcpy(const_cast<unconst_type*>(dst), src, size_bytes);
      }
      else
      {
         // we have a subarray, potentially with stride so we need to use a processor
         auto op_cpy = [&](T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements) {
            // @TODO add the BLAS copy
            NLL_FAST_ASSERT(!std::is_const<T>::value, "type is CONST!");
            details::copy_naive(const_cast<unconst_type*>(y_pointer), y_stride, x_pointer, x_stride, nb_elements);
         };
         iterate_memory_constmemory(*this, other, op_cpy);
      }
   }

public:
   Memory_gpu_cuda(const Memory_gpu_cuda& other)
   {
      _deepCopy(other);
   }

   Memory_gpu_cuda& operator=(const Memory_gpu_cuda& other)
   {
      _deepCopy(other);
      return *this;
   }

   Memory_gpu_cuda(Memory_gpu_cuda&& other)
   {
      // make sure we don't have non-moveable memory
      if (details::MemoryMoveable<Memory_gpu_cuda>::can_move(other._data, other._linearSize()))
      {
         _moveCopy(std::forward<Memory_gpu_cuda>(other));
      }
      else {
         _deepCopy(other);
      }
   }

   Memory_gpu_cuda& operator=(Memory_gpu_cuda&& other)
   {
      // make sure we don't have non-moveable memory
      if (details::MemoryMoveable<Memory_gpu_cuda>::can_move(other._data, other._linearSize()))
      {
         _moveCopy(std::forward<Memory_gpu_cuda>(other));
      }
      else {
         _deepCopy(other);
      }
      return *this;
   }

   ~Memory_gpu_cuda()
   {
      _deallocateSlices();
   }

   const T* at(const index_type& index) const
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   T* at(const index_type& index)
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   const index_type& shape() const
   {
      return _shape;
   }

   const index_mapper& getIndexMapper() const
   {
      return _indexMapper;
   }

   const allocator_type& getAllocator() const
   {
      return _allocator;
   }

private:
   // arrange py decreasing size order to help with the structure packing
   index_mapper     _indexMapper;
   T*               _data = nullptr;
   Memory_gpu_cuda* _sharedView = nullptr; /// the original array
   index_type       _shape;
   allocator_type   _allocator;
   bool             _dataAllocated = true;
};

DECLARE_NAMESPACE_NLL_END

#endif