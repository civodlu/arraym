#pragma once

DECLARE_NAMESPACE_NLL

/**
@brief Memory composed of multi-slices

Value based semantics, except when using sub-memory blocks which keeps a reference of the memory.

When a sub-block is created, a <SharedView> structure is used to prevent memory deallocation even though the source block is out of scope

index_mapper::Z_INDEX defines the indexes where the slices are created, all other dimensions are stored in a contiguous array
*/
template <class T, size_t N, class IndexMapper = IndexMapper_multislice<N, N - 1>, class Allocator = std::allocator<T>>
class Memory_multislice : public memory_layout_multislice_z
{
public:
   using Vectorui = core::StaticVector<ui32, N>;
   using allocator_type = Allocator;
   using allocator_trait = std::allocator_traits<allocator_type>;
   using index_mapper = IndexMapper;
   static const size_t Z_INDEX = index_mapper::Z_INDEX; /// this is the index where the slices will be created (i.e., all others will be in contiguous memory)

   template <class T>
   class diterator_t : public std::iterator < std::random_access_iterator_tag, T >
   {
      friend class diterator_t < const T >;
      typedef std::iterator<std::random_access_iterator_tag, T> Base;

      typedef typename Base::difference_type    difference_type;
      typedef typename Base::reference          reference;
      typedef typename Base::pointer            pointer;

   public:
      diterator_t() : _p( nullptr ), _slices( nullptr )
      {}

      diterator_t( ui32 slice, ui32 offset, size_t stride, T** slices ) : _slice( slice ), _offset( offset ), _stride( stride ), _slices( slices )
      {
         _p = slices[ slice ] + offset;
      }

      diterator_t( const diterator_t<typename std::remove_const<T>::type>& other ) : diterator_t( other._slice, other._offset, other._stride, other._slices )
      {
         _p = other._p;
      }

      diterator_t& operator++( )
      {
         if ( _stride == 0 )
         {
            _p = _slices[ ++_slice ] + _offset;
         } else
         {
            _p += _stride;
         }

         return *this;
      }

      diterator_t& add( int step = 1 )
      {
         if ( _stride == 0 )
         {
            _slice += step;
            _p = _slices[ _slice ] + _offset;
         } else
         {
            _p += _stride * step;
         }

         return *this;
      }

      bool operator==( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         return rhs._p == _p;
      }

      bool operator!=( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         return rhs._p != _p;
      }

      difference_type operator-( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         if ( _stride == 0 )
         {
            NLL_FAST_ASSERT( _offset == rhs._offset, "non matching <D> iterator" );
            return _slice - rhs._slice;
         }
         return ( _p - rhs._p ) / _stride;
      }

      reference operator*( )
      {
         return *_p;
      }

   private:
      ui32    _slice;
      ui32    _offset;
      size_t  _stride;
      T**     _slices;
      T*      _p;
   };

   using diterator = diterator_t<T>;
   using const_diterator = diterator_t<const T>;

   Memory_multislice( const allocator_type& allocator = allocator_type() ) :
      _allocator( allocator )
   {}

   /// New memory block
   Memory_multislice( const Vectorui& shape, T default_value = T(), const allocator_type& allocator = allocator_type() ) :
      _shape( shape ), _allocator( allocator )
   {
      _indexMapper.init( 0, shape );
      _allocateSlices( default_value, _inSliceSize() );
   }

   /**
   @param slices pre-existing slices. if <slicesAllocated>, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, <allocator> will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_multislice( const Vectorui& shape, const std::vector<T*>& slices, const allocator_type& allocator = allocator_type(), bool slicesAllocated = false ) :
      _shape( size ), _allocator( allocator )
   {
      _indexMapper.init( 0, shape );
      _slices = slices;
      _slicesAllocated = slicesAllocated;
   }

   /**
   @brief reference an existing sub-memory block
   @param strides the stride (spacing between data elements) to be used to access the <ref>. (i.e., this doesn't depend on the
   stride of <ref> itself
   @param shape the size of the actual memory
   @param min_index the index in <ref> to be used as the first data element
   */
   Memory_multislice( Memory_multislice& ref, const Vectorui& min_index, const Vectorui& shape, const Vectorui& strides ) :
      _shape( shape )
   {
      _slices.resize( shape[ Z_INDEX ] );
      for ( size_t s = 0; s < shape[ Z_INDEX ]; ++s )
      {
         const size_t slice = min_index[ Z_INDEX ] + s * strides[ Z_INDEX ];
         _slices[ s ] = ref._getSlices()[ slice ];
      }
      _slicesAllocated = false; // we create a reference on existing slices
      _indexMapper = ref.getIndexMapper().submap( min_index, shape, strides );

      //
      // handle memory references
      //

      if ( ref._sharedView.get() )
      {
         // there is already a reference, just increase the refcount
         _sharedView = ref._sharedView;
      } else
      {
         // since we are creating a reference, create a shared state holding the original memory and not to be destroyed until all references
         // are destroyed
         _sharedView = std::shared_ptr<SharedView>( new SharedView{ _slices, _allocator, ref._inSliceSize() } );
         ref._sharedView = _sharedView;
         ref._slicesAllocated = false; // now delegate the memory management to the shared ptr
      }
   }

   diterator beginDim( ui32 dim, const Vectorui& indexN )
   {
      return diterator( indexN[ Z_INDEX ], _indexMapper.offset( indexN ), _indexMapper._getPhysicalStrides()[ dim ], &_slices[ 0 ] );
   }

   const_diterator beginDim( ui32 dim, const Vectorui& indexN ) const
   {
      return const_diterator( indexN[ Z_INDEX ], _indexMapper.offset( indexN ), _indexMapper._getPhysicalStrides()[ dim ], &_slices[ 0 ] );
   }

   diterator endDim( ui32 dim, const Vectorui& indexN )
   {
      Vectorui index_cpy = indexN;
      index_cpy[ dim ] = this->_shape[ dim ];
      return diterator( index_cpy[ Z_INDEX ], _indexMapper.offset( index_cpy ), _indexMapper._getPhysicalStrides()[ dim ], &_slices[ 0 ] );
   }

   const_diterator endDim( ui32 dim, const Vectorui& indexN ) const
   {
      Vectorui index_cpy = indexN;
      index_cpy[ dim ] = this->_shape[ dim ];
      return const_diterator( index_cpy[ Z_INDEX ], _indexMapper.offset( index_cpy ), _indexMapper._getPhysicalStrides()[ dim ], &_slices[ 0 ] );
   }

private:
   ui32 _inSliceSize() const
   {
      ui32 size = 1;
      for ( size_t n = 0; n < N; ++n )
      {
         if ( n != Z_INDEX )
         {
            size *= _shape[ n ];
         }
      }
      return size;
   }

   void _allocateSlices( T default_value, ui32 in_slice_size )
   {
      _slices.resize( _shape[ Z_INDEX ] );
      for ( size_t n = 0; n < _slices.size(); ++n )
      {
         auto p = allocator_trait::allocate( _allocator, in_slice_size );
         for ( size_t nn = 0; nn < in_slice_size; ++nn )
         {
            allocator_trait::construct( _allocator, p + nn, default_value );
         }
         _slices[ n ] = p;
      }
   }

   void _deallocateSlices()
   {
      if ( _slicesAllocated )
      {
         const auto in_slice_size = _inSliceSize();
         for ( size_t n = 0; n < _slices.size(); ++n )
         {
            auto p = _slices[ n ];
            for ( size_t nn = 0; nn < in_slice_size; ++nn )
            {
               allocator_trait::destroy( _allocator, p + nn );
            }
            allocator_trait::deallocate( _allocator, p, in_slice_size );
         }
      }

      _sharedView.reset();
   }

   void _deepCopy( const Memory_multislice& other )
   {
      //
      // TODO: we do NOT want to copy the full base memory, we SHOULD revert to stride = 1, 1, 1
      // use a <processor> here!
      //

      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _indexMapper = other._indexMapper;
      _shape = other._shape;
      _allocator = other._allocator;
      _slicesAllocated = true;

      // now deep copy...
      ui32 in_slice_size = 0;
      if ( other._sharedView.get() )
      {
         // do not use the actual size of the reference, it may be a sub-memory!
         in_slice_size = other._sharedView->inSliceSize;
      } else
      {
         // there is no reference so it can't be a sub-memory and _size is its actual size
         in_slice_size = _inSliceSize();
      }
      const auto size_per_slice_bytes = sizeof( T ) * in_slice_size;
      _allocateSlices( T(), in_slice_size );
      for ( size_t n = 0; n < _shape[ Z_INDEX ]; ++n )
      {
         static_assert( std::is_standard_layout<T>::value, "must have standard layout!" );
         const auto src = other._slices[ n ];
         const auto dst = _slices[ n ];
         memcpy( dst, src, size_per_slice_bytes );
      }
   }

   void _moveCopy( Memory_multislice&& other )
   {
      if ( this != &other )
      {
         _indexMapper = other._indexMapper;
         _shape = other._shape;
         _allocator = other._allocator;

         _slicesAllocated = other._slicesAllocated;
         other._slicesAllocated = false;

         _slices = std::move( other._slices );
         _sharedView = std::move( other._sharedView );
      }
   }

public:
   Memory_multislice( const Memory_multislice& other )
   {
      _deepCopy( other );
   }

   Memory_multislice& operator=( const Memory_multislice& other )
   {
      _deepCopy( other );
      return *this;
   }


   Memory_multislice( Memory_multislice&& other )
   {
      _moveCopy( std::forward<Memory_multislice>( other ) );
   }

   Memory_multislice& operator=( Memory_multislice&& other )
   {
      _moveCopy( std::forward<Memory_multislice>( other ) );
      return *this;
   }

   ~Memory_multislice()
   {
      _deallocateSlices();
   }

   const T* at( const Vectorui& index ) const
   {
      const auto offset = _indexMapper.offset( index );
      const auto p = _slices[ index[ Z_INDEX ] ];
      return p + offset;
   }

   T* at( const Vectorui& index )
   {
      const auto offset = _indexMapper.offset( index );
      const auto p = _slices[ index[ Z_INDEX ] ];
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

   const Vectorui& getShape() const
   {
      return _shape;
   }

   ui32 getOrigin() const
   {
      return _origin;
   }

   const IndexMapper& getIndexMapper() const
   {
      return _indexMapper;
   }

private:
   struct SharedView
   {
      std::vector<T*>  slices;
      allocator_type   allocator;
      ui32             inSliceSize;

      ~SharedView()
      {
         for ( auto ptr : slices )
         {
            for ( size_t nn = 0; nn < inSliceSize; ++nn )
            {
               allocator_trait::destroy( allocator, ptr + nn );
            }
            allocator_trait::deallocate( allocator, ptr, inSliceSize );
         }
      }
   };

   IndexMapper      _indexMapper;
   Vectorui         _shape;
   std::vector<T*>  _slices;
   allocator_type   _allocator;
   bool             _slicesAllocated = true;

   std::shared_ptr<SharedView> _sharedView;  /// when we reference an existing view, we must shared the memory...
};

/**
@brief Memory composed of multi-slices

Value based semantics, except when using sub-memory blocks which keeps a reference of the memory.

When a sub-block is created, a <SharedView> structure is used to prevent memory deallocation even though the source block is out of scope

index_mapper::Z_INDEX defines the indexes where the slices are created, all other dimensions are stored in a contiguous array
*/
template <class T, size_t N, class IndexMapper = IndexMapper_contiguous<N>, class Allocator = std::allocator<T>>
class Memory_contiguous : public memory_layout_contiguous
{
public:
   using Vectorui = core::StaticVector<ui32, N>;
   using allocator_type = Allocator;
   using allocator_trait = std::allocator_traits<allocator_type>;
   using index_mapper = IndexMapper;

   template <class T>
   class diterator_t : public std::iterator < std::random_access_iterator_tag, T >
   {
      friend class diterator_t < const T >;
      typedef std::iterator<std::random_access_iterator_tag, T> Base;

      typedef typename Base::difference_type    difference_type;
      typedef typename Base::reference          reference;
      typedef typename Base::pointer            pointer;

   public:
      diterator_t() : _p( nullptr )
      {}

      diterator_t( T* p, size_t stride ) : _p( p ), _stride( stride )
      {}

      diterator_t( const diterator_t<typename std::remove_const<T>::type>& other ) : diterator_t( other._p, other._stride )
      {}

      diterator_t& operator++( )
      {
         _p += _stride;
         return *this;
      }

      diterator_t& add( int step = 1 )
      {
         _p += _stride * step;
         return *this;
      }

      bool operator==( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         return rhs._p == _p;
      }

      bool operator!=( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         return rhs._p != _p;
      }

      difference_type operator-( const diterator_t& rhs ) const
      {
         NLL_FAST_ASSERT( _stride == rhs._stride, "non matching <D> iterator" );
         return ( _p - rhs._p ) / _stride;
      }

      reference operator*( )
      {
         return *_p;
      }

   private:
      pointer _p;
      size_t  _stride;
   };

   using diterator = diterator_t<T>;
   using const_diterator = diterator_t<const T>;


   Memory_contiguous( const allocator_type& allocator = allocator_type() ) :
      _allocator( allocator )
   {}

   /// New memory block
   Memory_contiguous( const Vectorui& shape, T default_value = T(), const allocator_type& allocator = allocator_type() ) :
      _shape( shape ), _allocator( allocator )
   {
      _indexMapper.init( 0, shape );
      _allocateSlices( default_value, _linearSize() );
   }

   /**
   @param slices pre-existing slices. if <slicesAllocated>, allocator will be used to deallocate the memory. Else the user is responsible
   for deallocation
   @param slicesAllocated if true, <allocator> will be used to deallocate the memory. Else the user is responsible for the slice's memory
   */
   Memory_contiguous( const Vectorui& shape, const T* data, const allocator_type& allocator = allocator_type(), bool dataAllocated = false ) :
      _shape( size ), _allocator( allocator )
   {
      _indexMapper.init( 0, shape );
      _data = data;
      _dataAllocated = dataAllocated;
   }

   /**
   @brief reference an existing sub-memory block
   @param strides the stride (spacing between data elements) to be used to access the <ref>. (i.e., this doesn't depend on the
   stride of <ref> itself
   @param shape the size of the actual memory
   @param min_index the index in <ref> to be used as the first data element
   */
   Memory_contiguous( Memory_contiguous& ref, const Vectorui& min_index, const Vectorui& shape, const Vectorui& strides ) :
      _shape( shape )
   {
      _data = ref._data;
      _dataAllocated = false; // we create a reference on existing slices
      _indexMapper = ref.getIndexMapper().submap( min_index, shape, strides );

      //
      // handle memory references
      //

      if ( ref._sharedView.get() )
      {
         // there is already a reference, just increase the refcount
         _sharedView = ref._sharedView;
      } else
      {
         // since we are creating a reference, create a shared state holding the original memory and not to be destroyed until all references
         // are destroyed
         _sharedView = std::shared_ptr<SharedView>( new SharedView{ _data, _allocator, ref._linearSize() } );
         ref._sharedView = _sharedView;
         ref._dataAllocated = false; // now delegate the memory management to the shared ptr
      }
   }

   diterator beginDim( ui32 dim, const Vectorui& indexN )
   {
      auto p = this->at( indexN );
      return diterator( p, _indexMapper._getPhysicalStrides()[ dim ] );
   }

   const_diterator beginDim( ui32 dim, const Vectorui& indexN ) const
   {
      auto p = this->at( indexN );
      return const_diterator( p, _indexMapper._getPhysicalStrides()[ dim ] );
   }

   diterator endDim( ui32 dim, const Vectorui& indexN )
   {
      Vectorui index_cpy = indexN;
      index_cpy[ dim ] = this->_shape[ dim ];

      auto p = this->at( index_cpy );
      return diterator( p, _indexMapper._getPhysicalStrides()[ dim ] );
   }

   const_diterator endDim( ui32 dim, const Vectorui& indexN ) const
   {
      Vectorui index_cpy = indexN;
      index_cpy[ dim ] = this->_shape[ dim ];

      auto p = this->at( index_cpy );
      return const_diterator( p, _indexMapper._getPhysicalStrides()[ dim ] );
   }

private:
   ui32 _linearSize() const
   {
      ui32 size = 1;
      for ( size_t n = 0; n < N; ++n )
      {
         size *= _shape[ n ];
      }
      return size;
   }

   void _allocateSlices( T default_value, ui32 linear_size )
   {
      auto p = allocator_trait::allocate( _allocator, linear_size );
      for ( size_t nn = 0; nn < linear_size; ++nn )
      {
         allocator_trait::construct( _allocator, p + nn, default_value );
      }
      _data = p;
   }

   void _deallocateSlices()
   {
      if ( _dataAllocated )
      {
         const auto linear_size = _linearSize();
         for ( size_t nn = 0; nn < linear_size; ++nn )
         {
            allocator_trait::destroy( _allocator, _data + nn );
         }
         allocator_trait::deallocate( _allocator, _data, linear_size );
      }

      _data = nullptr;
      _sharedView.reset();
   }

   void _moveCopy( Memory_contiguous&& other )
   {
      if ( this != &other )
      {
         _indexMapper = other._indexMapper;
         _shape = other._shape;
         _allocator = other._allocator;

         _dataAllocated = other._dataAllocated;
         other._dataAllocated = false;

         _data = other._data;
         other._data = nullptr;

         _sharedView = std::move( other._sharedView );
      }
   }

   void _deepCopy( const Memory_contiguous& other )
   {
      //
      // TODO: we do NOT want to copy the full base memory, we SHOULD revert to stride = 1, 1, 1
      // use a <processor> here!
      //

      _deallocateSlices(); // if any memory is allocated or referenced, they are not needed anymore

      _indexMapper = other._indexMapper;
      _shape = other._shape;
      _allocator = other._allocator;
      _dataAllocated = true;

      // now deep copy...
      ui32 linear_size = 0;
      if ( other._sharedView.get() )
      {
         // do not use the actual size of the reference, it may be a sub-memory!
         linear_size = other._sharedView->linear_size;
      } else
      {
         // there is no reference so it can't be a sub-memory and _size is its actual size
         linear_size = _linearSize();
      }
      const auto size_bytes = sizeof( T ) * linear_size;
      _allocateSlices( T(), linear_size );

      static_assert( std::is_standard_layout<T>::value, "must have standard layout!" );
      const auto src = other._data;
      const auto dst = _data;
      memcpy( dst, src, size_bytes );
   }

public:
   Memory_contiguous( const Memory_contiguous& other )
   {
      _deepCopy( other );
   }

   Memory_contiguous& operator=( const Memory_contiguous& other )
   {
      _deepCopy( other );
      return *this;
   }


   Memory_contiguous( Memory_contiguous&& other )
   {
      _moveCopy( std::forward<Memory_contiguous>( other ) );
   }

   Memory_contiguous& operator=( Memory_contiguous&& other )
   {
      _moveCopy( std::forward<Memory_contiguous>( other ) );
      return *this;
   }

   ~Memory_contiguous()
   {
      _deallocateSlices();
   }

   const T* at( const Vectorui& index ) const
   {
      const auto offset = _indexMapper.offset( index );
      return _data + offset;
   }

   T* at( const Vectorui& index )
   {
      const auto offset = _indexMapper.offset( index );
      return _data + offset;
   }

   const Vectorui& getShape() const
   {
      return _shape;
   }

   ui32 getOrigin() const
   {
      return _origin;
   }

   const IndexMapper& getIndexMapper() const
   {
      return _indexMapper;
   }

private:
   struct SharedView
   {
      T*               data;
      allocator_type   allocator;
      ui32             linear_size;

      ~SharedView()
      {
         for ( size_t nn = 0; nn < linear_size; ++nn )
         {
            allocator_trait::destroy( allocator, data + nn );
         }
         allocator_trait::deallocate( allocator, data, linear_size );
         data = nullptr;
      }
   };

   IndexMapper      _indexMapper;
   Vectorui         _shape;
   T*               _data = nullptr;
   allocator_type   _allocator;
   bool             _dataAllocated = true;

   std::shared_ptr<SharedView> _sharedView;  /// when we reference an existing view, we must shared the memory...
};

template <class T, int N, class Allocator = std::allocator<T>>
using Memory_contiguous_row_major = Memory_contiguous < T, N, IndexMapper_contiguous_row_major<N>, Allocator >;

DECLARE_NAMESPACE_END