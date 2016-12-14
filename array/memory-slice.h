#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Memory composed of multi-slices

Value based semantics, except when using sub-memory blocks which keeps a reference of the memory.

A sub-block must NOT be accessed when the original memory has been destroyed: dangling pointer!

index_mapper::Z_INDEX defines the indexes where the slices are created, all other dimensions are stored in a contiguous array
*/
template <class T, size_t N, class IndexMapper = IndexMapper_multislice<N, N - 1>, class Allocator = std::allocator<T>>
class Memory_multislice : public memory_layout_multislice_z
{
public:
   using index_type      = StaticVector<ui32, N>;
   using allocator_type  = Allocator;
   using allocator_trait = std::allocator_traits<allocator_type>;
   using index_mapper    = IndexMapper;
   using pointer_type    = T*;
   using value_type      = T;
   using Memory          = Memory_multislice<T, N, IndexMapper, Allocator>;

   static const size_t Z_INDEX = index_mapper::Z_INDEX; /// this is the index where the slices will be created (i.e., all others will be in contiguous memory)
   static const size_t RANK    = N;

   /**
   @brief Rebind the memory with another type
    */
   template <class T2>
   struct rebind
   {
      using unconst_type = typename std::remove_const<T2>::type;

      // do NOT use the const in the allocator: this is underfined and won't compile for GCC/Clang
      using other = Memory_multislice<T2, N, IndexMapper, typename Allocator::template rebind<unconst_type>::other>;
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
      diterator_t() : _p(nullptr), _slices(nullptr)
      {
      }

      diterator_t(ui32 slice, ui32 offset, size_t stride, TT** slices) : _slice(slice), _offset(offset), _stride(stride), _slices(slices)
      {
         _p = slices[slice] + offset;
      }

      diterator_t(const diterator_t<typename std::remove_const<TT>::type>& other) : diterator_t(other._slice, other._offset, other._stride, other._slices)
      {
         _p = other._p;
      }

      diterator_t& operator++()
      {
         if (_stride == 0)
         {
            _p = _slices[++_slice] + _offset;
         }
         else
         {
            _p += _stride;
         }

         return *this;
      }

      diterator_t& add(int step = 1)
      {
         if (_stride == 0)
         {
            _slice += step;
            _p = _slices[_slice] + _offset;
         }
         else
         {
            _p += _stride * step;
         }

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
         if (_stride == 0)
         {
            NLL_FAST_ASSERT(_offset == rhs._offset, "non matching <D> iterator");
            return _slice - rhs._slice;
         }
         return (_p - rhs._p) / _stride;
      }

      reference operator*()
      {
         return *_p;
      }

   private:
      ui32 _slice;
      ui32 _offset;
      size_t _stride;
      TT** _slices;
      TT* _p;
   };

   using diterator       = diterator_t<T>;
   using const_diterator = diterator_t<const T>;

   Memory_multislice(const allocator_type& allocator = allocator_type()) : _allocator(allocator)
   {
   }

   /// New memory block
   Memory_multislice(const index_type& shape, T default_value = T(), const allocator_type& allocator = allocator_type()) : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _allocateSlices(default_value, _inSliceSize());
   }

   /**
   @param slices pre-existing slices. if @p slicesAllocated, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, @p allocator will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_multislice(const index_type& shape, const std::vector<T*>& slices, const allocator_type& allocator = allocator_type(), bool slicesAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _slices          = slices;
      _slicesAllocated = slicesAllocated;
   }

   Memory_multislice(const index_type& shape, const std::vector<T*>& slices, const index_type& physicalStrides,
                     const allocator_type& allocator = allocator_type(), bool slicesAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(physicalStrides);
      _slices          = slices;
      _slicesAllocated = slicesAllocated;
   }

   /**
   @brief reference an existing sub-memory block
   @param strides the stride (spacing between data elements) to be used to access the @p ref. (i.e., this doesn't depend on the
   stride of @p ref itself
   @param shape the size of the actual memory
   @param min_index the index in @p ref to be used as the first data element
   */
   Memory_multislice(Memory_multislice& ref, const index_type& min_index, const index_type& shape, const index_type& strides) : _shape(shape)
   {
      _slices.resize(shape[Z_INDEX]);
      for (size_t s = 0; s < shape[Z_INDEX]; ++s)
      {
         const size_t slice = min_index[Z_INDEX] + s * strides[Z_INDEX];
         _slices[s]         = ref._getSlices()[slice];
      }
      _slicesAllocated = false; // we create a reference on existing slices
      _indexMapper     = ref.getIndexMapper().submap(min_index, shape, strides);

      //
      // handle memory references
      //

      if (ref._sharedView)
      {
         // there is already a reference, just increase the refcount
         _sharedView = ref._sharedView;
      }
      else
      {
         // create a reference
         _sharedView = &ref;
      }
   }

   ConstMemory asConst() const
   {
      return ConstMemory(shape(), reinterpret_cast<const std::vector<const T*>&>(_slices), getIndexMapper()._getPhysicalStrides());
   }

   const allocator_type& getAllocator() const
   {
      return _allocator;
   }

   diterator beginDim(ui32 dim, const index_type& indexN)
   {
      return diterator(indexN[Z_INDEX], _indexMapper.offset(indexN), _indexMapper._getPhysicalStrides()[dim], &_slices[0]);
   }

   const_diterator beginDim(ui32 dim, const index_type& indexN) const
   {
      return const_diterator(indexN[Z_INDEX], _indexMapper.offset(indexN), _indexMapper._getPhysicalStrides()[dim], &_slices[0]);
   }

   diterator endDim(ui32 dim, const index_type& indexN)
   {
      index_type index_cpy = indexN;
      index_cpy[dim]       = this->_shape[dim];
      return diterator(index_cpy[Z_INDEX], _indexMapper.offset(index_cpy), _indexMapper._getPhysicalStrides()[dim], &_slices[0]);
   }

   const_diterator endDim(ui32 dim, const index_type& indexN) const
   {
      index_type index_cpy = indexN;
      index_cpy[dim]       = this->_shape[dim];
      return const_diterator(index_cpy[Z_INDEX], _indexMapper.offset(index_cpy), _indexMapper._getPhysicalStrides()[dim], &_slices[0]);
   }

private:
   // general case: we are keeping the Z_INDEX so we are still a slice based memory. Z_INDEX will be decreased
   // Note: this is OK to decrease Z_INDEX when it is set to be the last dimension. In a more general setting,
   //       this is not correct.
   struct SliceImpl_notz
   {
      static_assert(Z_INDEX == N - 1, "@TODO not handled yet. Need a mechanism to reassign Z_INDEX");
      using other = Memory_multislice<T, N - 1, typename IndexMapper::template rebind<N - 1, Z_INDEX - 1>::other, allocator_type>;

      template <size_t slice_dim>
      static other slice(const Memory_multislice& array, const index_type& index)
      {
         typename other::index_type shape;
         typename other::index_type physicalStride;
         size_t current_index = 0;
         for (size_t n = 0; n < N; ++n)
         {
            if (n != slice_dim)
            {
               shape[current_index]          = array.shape()[n];
               physicalStride[current_index] = array.getIndexMapper()._getPhysicalStrides()[n];
               ++current_index;
            }
         }

         const size_t nb_slices = array.shape()[Z_INDEX];
         std::vector<T*> slices(nb_slices);
         index_type origin;
         origin[slice_dim] = index[slice_dim];
         for (ui32 n = 0; n < nb_slices; ++n)
         {
            origin[Z_INDEX] = n;
            slices[n]       = const_cast<T*>(array.at(origin));
         }

         return other(shape, slices, physicalStride, array.getAllocator(), false);
      }
   };

   // specific case: we slice a dimension that is Z_INDEX so we need to
   // keep a single slice
   struct SliceImpl_z
   {
      using index_mapper = IndexMapper_contiguous<N - 1, details::Mapper_stride_row_major<N - 1>>;
      using other        = Memory_contiguous<T, N - 1, index_mapper, allocator_type>;

      template <size_t slice_dim>
      static other slice(const Memory_multislice& array, const index_type& index)
      {
         // we start at the beginning of the slice
         index_type index_slice;
         index_slice[slice_dim] = index[slice_dim];
         T* ptr                 = const_cast<T*>(array.at(index_slice));

         typename other::index_type shape;
         typename other::index_type physicalStride;
         size_t current_index = 0;
         for (size_t n = 0; n < N; ++n)
         {
            if (n != Z_INDEX)
            {
               shape[current_index]          = array.shape()[n];
               physicalStride[current_index] = array.getIndexMapper()._getPhysicalStrides()[n];
               ++current_index;
            }
         }

         return other(shape, ptr, physicalStride, array.getAllocator(), false);
      }
   };

   template <size_t slice_dim>
   using SliceImpl = std::conditional<slice_dim == Z_INDEX, SliceImpl_z, SliceImpl_notz>;


public:
   template <class T2, size_t N2>
   struct rebind_type_dim
   {
      // @TODO adaptor for Array slicing type. Refactor
      using other = typename SliceImpl<N2>::type::other;
   };

public:
   /*
   template <size_t dimension>
   using slice_type = typename SliceImpl<dimension>::type::other;
   */

   /**
   @brief Slice the memory such that we keep only the slice along dimension @p dimension passing through @p point

   Create a reference of this object, so do NOT destroy the memory while using the sliced mempory

   Workaround for VS2013 internal compiler bug with
   "SliceImpl = std::conditional<slice_dim == Z_INDEX, SliceImpl_z, SliceImpl_notz>::type;"
   */
   template <size_t dimension>
   typename SliceImpl<dimension>::type::other slice(const index_type& point) const
   {
      using Impl = typename SliceImpl<dimension>::type;
      return Impl::template slice<dimension>(*this, point);
   }

private:
   ui32 _inSliceSize() const
   {
      ui32 size = 1;
      for (size_t n = 0; n < N; ++n)
      {
         if (n != Z_INDEX)
         {
            size *= _shape[n];
         }
      }
      return size;
   }

   void _allocateSlices(T default_value, ui32 in_slice_size)
   {
      _slices.resize(_shape[Z_INDEX]);
      for (size_t n = 0; n < _slices.size(); ++n)
      {
         auto p = allocator_trait::allocate(_allocator, in_slice_size);
         for (size_t nn = 0; nn < in_slice_size; ++nn)
         {
            allocator_trait::construct(_allocator, p + nn, default_value);
         }
         _slices[n] = p;
      }
   }

   void _deallocateSlices()
   {
      if (_slicesAllocated)
      {
         const auto in_slice_size = _inSliceSize();
         for (size_t n = 0; n < _slices.size(); ++n)
         {
            auto p = _slices[n];
            for (size_t nn = 0; nn < in_slice_size; ++nn)
            {
               allocator_trait::destroy(_allocator, p + nn);
            }
            // handle the const T* case for const arrays
            using unconst_value = typename std::remove_cv<T>::type;
            allocator_trait::deallocate(_allocator, const_cast<unconst_value*>(p), in_slice_size);
         }
      }

      _sharedView = nullptr;
   }

   void _deepCopy(const Memory_multislice& other)
   {
      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _shape = other._shape;
      _indexMapper.init(0, _shape);
      _allocator       = other._allocator;
      _slicesAllocated = true;

      const ui32 this_inSliceSize  = _inSliceSize();
      const ui32 other_inSliceSize = other._sharedView ? other._sharedView->_inSliceSize() : other._inSliceSize();

      const auto size_per_slice_bytes = sizeof(T) * this_inSliceSize;
      _allocateSlices(T(), this_inSliceSize);
      if (this_inSliceSize == other_inSliceSize &&
          details::IsMemoryFullyContiguous<Memory>::value(
              other, std::integral_constant<bool, true>())) // we want fully contiguous slice! Else if the stride is not (1, ..., 1) the memcpy would be invalid
      {
         // same size so there was no sub-array: we can directly copy the memory in a single block
         for (size_t n = 0; n < _shape[Z_INDEX]; ++n)
         {
            static_assert(std::is_standard_layout<T>::value, "must have standard layout!");
            const auto src = other._slices[n];
            const auto dst = _slices[n];
            memcpy(dst, src, size_per_slice_bytes);
         }
      }
      else
      {
         // we have a subarray, potentially with stride so we need to use a processor
         auto op_cpy = [&](T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements) {
            /// @TODO add the BLAS copy
            details::copy_naive(y_pointer, y_stride, x_pointer, x_stride, nb_elements);
         };
         iterate_memory_constmemory(*this, other, op_cpy);
      }
   }

   void _moveCopy(Memory_multislice&& other)
   {
      if (this != &other)
      {
         _indexMapper = other._indexMapper;
         _shape       = other._shape;
         _allocator   = other._allocator;

         _slicesAllocated       = other._slicesAllocated;
         other._slicesAllocated = false;

         _slices     = std::move(other._slices);
         _sharedView = other._sharedView;
      }
   }

public:
   Memory_multislice(const Memory_multislice& other)
   {
      _deepCopy(other);
   }

   Memory_multislice& operator=(const Memory_multislice& other)
   {
      _deepCopy(other);
      return *this;
   }

   Memory_multislice(Memory_multislice&& other)
   {
      _moveCopy(std::forward<Memory_multislice>(other));
   }

   Memory_multislice& operator=(Memory_multislice&& other)
   {
      _moveCopy(std::forward<Memory_multislice>(other));
      return *this;
   }

   ~Memory_multislice()
   {
      _deallocateSlices();
   }

   const T* at(const index_type& index) const
   {
      const auto offset = _indexMapper.offset(index);
      const auto p      = _slices[index[Z_INDEX]];
      return p + offset;
   }

   T* at(const index_type& index)
   {
      const auto offset = _indexMapper.offset(index);
      const auto p      = _slices[index[Z_INDEX]];
      return p + offset;
   }

   //
   // for debug purposes only!
   const std::vector<T*>& _getSlices() const
   {
      return _slices;
   }
   //
   //

   const index_type& shape() const
   {
      return _shape;
   }

   const IndexMapper& getIndexMapper() const
   {
      return _indexMapper;
   }

private:
   IndexMapper _indexMapper;
   index_type _shape;
   std::vector<T*> _slices;
   allocator_type _allocator;
   bool _slicesAllocated = true;

   Memory_multislice* _sharedView = nullptr; /// the original array
};

DECLARE_NAMESPACE_END
