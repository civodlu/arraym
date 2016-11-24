#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Memory composed of multi-slices

Value based semantics, except when using sub-memory blocks which keeps a reference of the memory.

A sub-block must NOT be accessed when the original memory has been destroyed: dangling pointer!
*/
template <class T, size_t N, class IndexMapper = IndexMapper_contiguous<N>, class Allocator = std::allocator<T>>
class Memory_contiguous : public memory_layout_contiguous
{
public:
   using Vectorui        = StaticVector<ui32, N>;
   using allocator_type  = Allocator;
   using allocator_trait = std::allocator_traits<allocator_type>;
   using index_mapper    = IndexMapper;

   template <class T2, size_t N2>
   struct rebind
   {
      // using other = int;
      using other = Memory_contiguous<T2, N2, typename IndexMapper::template rebind<N2>::other, typename Allocator::template rebind<T2>::other>;
   };

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
   Memory_contiguous(const Vectorui& shape, T default_value = T(), const allocator_type& allocator = allocator_type()) : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _allocateSlices(default_value, _linearSize());
   }

   /**
   @param slices pre-existing slices. if <slicesAllocated>, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, <allocator> will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_contiguous(const Vectorui& shape, T* data, const allocator_type& allocator = allocator_type(), bool dataAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(0, shape);
      _data          = data;
      _dataAllocated = dataAllocated;
   }

   /**
   @param slices pre-existing slices. if <slicesAllocated>, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, <allocator> will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_contiguous(const Vectorui& shape, T* data, const Vectorui& physicalStrides, const allocator_type& allocator = allocator_type(),
                     bool dataAllocated = false)
       : _shape(shape), _allocator(allocator)
   {
      _indexMapper.init(physicalStrides);
      _data          = data;
      _dataAllocated = dataAllocated;
   }

   /**
   @brief Slice the memory such that we keep only the slice along dimension <dimension> passing through <point>

   Create a reference of <this>, so do NOT destroy the memory while using the sliced mempory
   */
   template <size_t dimension>
   typename rebind<T, N - 1>::other slice(const Vectorui& index) const
   {
      // we start at the beginning of the slice
      Vectorui index_slice;
      index_slice[dimension] = index[dimension];
      T* ptr                 = const_cast<T*>(this->at(index_slice));

      using Other = typename rebind<T, N - 1>::other;
      Other memory;
      memory._indexMapper   = _indexMapper.slice<dimension>(index_slice);
      memory._data          = const_cast<T*>(ptr);
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
   @param strides the stride (spacing between data elements) to be used to access the <ref>. (i.e., this doesn't depend on the
   stride of <ref> itself
   @param shape the size of the actual memory
   @param min_index the index in <ref> to be used as the first data element
   */
   Memory_contiguous(Memory_contiguous& ref, const Vectorui& min_index, const Vectorui& shape, const Vectorui& strides) : _shape(shape)
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

   diterator beginDim(ui32 dim, const Vectorui& indexN)
   {
      auto p = this->at(indexN);
      return diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   const_diterator beginDim(ui32 dim, const Vectorui& indexN) const
   {
      auto p = this->at(indexN);
      return const_diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   diterator endDim(ui32 dim, const Vectorui& indexN)
   {
      Vectorui index_cpy = indexN;
      index_cpy[dim]     = this->_shape[dim];

      auto p = this->at(index_cpy);
      return diterator(p, _indexMapper._getPhysicalStrides()[dim]);
   }

   const_diterator endDim(ui32 dim, const Vectorui& indexN) const
   {
      Vectorui index_cpy = indexN;
      index_cpy[dim]     = this->_shape[dim];

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
      auto p = allocator_trait::allocate(_allocator, linear_size);
      for (size_t nn = 0; nn < linear_size; ++nn)
      {
         allocator_trait::construct(_allocator, p + nn, default_value);
      }
      _data = p;
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
         allocator_trait::deallocate(_allocator, _data, linear_size);
      }

      _data       = nullptr;
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
         other._data = nullptr;

         _sharedView = other._sharedView;
      }
   }

   void _deepCopy(const Memory_contiguous& other)
   {
      //
      // TODO: we do NOT want to copy the full base memory, we SHOULD revert to stride = 1, 1, 1
      // use a <processor> here!
      //

      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _indexMapper   = other._indexMapper;
      _shape         = other._shape;
      _allocator     = other._allocator;
      _dataAllocated = true;

      // now deep copy...
      ui32 linear_size = 0;
      if (other._sharedView)
      {
         // do not use the actual size of the reference, it may be a sub-memory!
         linear_size = other._sharedView->_linearSize();
      }
      else
      {
         // there is no reference so it can't be a sub-memory and _size is its actual size
         linear_size = _linearSize();
      }
      const auto size_bytes = sizeof(T) * linear_size;
      _allocateSlices(T(), linear_size);

      static_assert(std::is_standard_layout<T>::value, "must have standard layout!");
      const auto src = other._data;
      const auto dst = _data;
      memcpy(dst, src, size_bytes);
   }

public:
   Memory_contiguous(const Memory_contiguous& other)
   {
      _deepCopy(other);
   }

   Memory_contiguous& operator=(const Memory_contiguous& other)
   {
      _deepCopy(other);
      return *this;
   }

   Memory_contiguous(Memory_contiguous&& other)
   {
      _moveCopy(std::forward<Memory_contiguous>(other));
   }

   Memory_contiguous& operator=(Memory_contiguous&& other)
   {
      _moveCopy(std::forward<Memory_contiguous>(other));
      return *this;
   }

   ~Memory_contiguous()
   {
      _deallocateSlices();
   }

   const T* at(const Vectorui& index) const
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   T* at(const Vectorui& index)
   {
      const auto offset = _indexMapper.offset(index);
      return _data + offset;
   }

   const Vectorui& shape() const
   {
      return _shape;
   }

   const IndexMapper& getIndexMapper() const
   {
      return _indexMapper;
   }

   const allocator_type& getAllocator() const
   {
      return _allocator;
   }

   //private:
   IndexMapper _indexMapper;
   Vectorui _shape;
   T* _data = nullptr;
   allocator_type _allocator;
   bool _dataAllocated = true;

   Memory_contiguous* _sharedView = nullptr; /// the original array
};

template <class T, int N, class Allocator = std::allocator<T>>
using Memory_contiguous_row_major = Memory_contiguous<T, N, IndexMapper_contiguous_row_major<N>, Allocator>;

DECLARE_NAMESPACE_END