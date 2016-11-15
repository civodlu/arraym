#pragma once

DECLARE_NAMESPACE_NLL

template <class T, int N, class Config>
class ArrayRef;

template <class T, int N, class ConfigT = ArrayTraitsConfig<T, N>>
class Array : public ArrayTraits<Array< T, N, ConfigT >, ConfigT>
{
public:
   using Config = ConfigT;
   using Memory = typename Config::Memory;
   using allocator_type = typename Config::allocator_type;
   using value_type = T;
   using array_type = Array < T, N, Config >;
   using array_type_ref = ArrayRef<T, N, Config>;
   using traits_type = ArrayTraits<Array< T, N, ConfigT >, ConfigT>;
   using pointer_type = T*;
   using reference_type = T&;
   using const_pointer_type = const T*;
   using index_type = core::StaticVector < ui32, N >;
   using diterator = typename Memory::diterator;
   using const_diterator = typename Memory::const_diterator;

   static const size_t RANK = N;

   template <class... Values>
   struct is_unpacked_arguments
   {
      static const bool value =
         sizeof...( Values ) == RANK &&
         core::is_same<Values...>::value &&
         std::is_integral<typename core::first<Values...>::type>::value &&
         !std::is_same<array_type, typename core::remove_cvr<typename core::first<Values...>::type>>::value;
   };

   // is this an example of: https://connect.microsoft.com/VisualStudio/feedback/details/1571800/false-positive-warning-c4520-multiple-default-constructors-specified
   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   Array( Values&&... values ) : _memory( index_type{ values... } )
   {}

   Array( const index_type& shape, T default_value = T(), const allocator_type& allocator = allocator_type() ) :
      _memory( shape, default_value, allocator )
   {}

   /**
   @brief create a shared sub-block
   */
   Array( Array& array, const index_type& origin, const index_type& shape, const index_type& stride ) :
      _memory( array._memory, origin, shape, stride )
   {
#ifndef NDEBUG
      for ( int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(origin[n] < array.shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(origin[n] + shape[n] <= array.shape()[n], "out of bounds!");
      }
#endif
   }

   Array( const allocator_type& allocator = allocator_type() ) :
      _memory( allocator )
   {}

   Array( const Memory& memory ) :
      _memory( memory )
   {}

   Array( Memory&& memory ) :
      _memory( std::forward<Memory>( memory ) )
   {}

   Array& operator=( const Array& other )
   {
      _copy( other );
      return *this;
   }

   Array( const Array& other )
   {
      _copy( other );
   }

   Array& operator=( Array&& other )
   {
      _move( std::forward<Array>( other ) );
      return *this;
   }

   Array( Array&& other )
   {
      _move( std::forward<Array>( other ) );
   }

   const index_type& shape() const
   {
      return _memory.getShape();
   }

   /**
   @brief Rank of the array
   */
   static int rank()
   {
      return N;
   }

   void write( std::ostream& f ) const
   {
      ensure( 0, "TODO implement" );
   }

   void read( std::istream& f )
   {
      ensure( 0, "TODO implement" );
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   reference_type operator()( const Values&... values )
   {
      index_type index = { values... };
      return operator()( index );
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   const reference_type operator()( const Values&... values ) const
   {
      index_type index = { values... };
      return operator()( index );
   }

   T& operator()( const index_type& index )
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(index[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return *_memory.at( index );
   }

   const T& operator()(const index_type& index) const
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(index[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return *_memory.at(index);
   }

   array_type_ref operator()(const index_type& min_index_inclusive, const index_type& max_index_inclusive)
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(min_index_inclusive[n] < this->shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(max_index_inclusive[n] < this->shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(min_index_inclusive[n] <= min_index_inclusive[n], "min > max!");
      }
#endif
      const auto size = max_index_inclusive - min_index_inclusive + 1;
      return array_type_ref(*this, min_index_inclusive, size, index_type(1));
   }

   diterator beginDim( ui32 dim, const index_type& indexN )
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.beginDim( dim, indexN );
   }

   const_diterator beginDim( ui32 dim, const index_type& indexN ) const
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.beginDim( dim, indexN );
   }

   diterator endDim( ui32 dim, const index_type& indexN )
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.endDim( dim, indexN );
   }

   const_diterator endDim( ui32 dim, const index_type& indexN ) const
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.endDim( dim, indexN );
   }

   const Memory& getMemory() const
   {
      return _memory;
   }

   bool isEmpty() const
   {
      for ( auto s : shape() )
      {
         if ( s == 0 )
         {
            return true;
         }
      }
      return false;
   }

private:
   void _move( array_type&& src )
   {
      if ( this != &src )
      {
         static_cast<traits_type&>( *this ) = std::move( src );
         _memory = std::move( src._memory );
      }
   }

   void _copy( const array_type& src )
   {
      static_cast<traits_type&>( *this ) = src;  // make sure th base class is copied
      _memory = src._memory;
   }

protected:
   Memory   _memory;
};


/**
@brief Default matrix type, following Fortran column-major style
@note If the default is changed, this means we need to update all the BLAS function calls having (LDA, LDB, ...) arguments
*/
template <class T, class Mapper = IndexMapper_contiguous_matrix_column_major, class Allocator = std::allocator<T> >
using Matrix = Array < T, 2,
   ArrayTraitsConfig<T, 2, Allocator, Memory_contiguous<T, 2, Mapper, Allocator>>
>;

template <class T, size_t N, class Allocator = std::allocator<T> >
using Array_row_major = Array < T, N,
   ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, IndexMapper_contiguous_row_major<N>, Allocator>>
>;

template <class T, size_t N, class Allocator = std::allocator<T> >
using Array_row_major_multislice = Array < T, N,
   ArrayTraitsConfig<T, N, Allocator, Memory_multislice<T, N, IndexMapper_multislice<N, N - 1>, Allocator>>
>;

template <class T, size_t N, class Allocator = std::allocator<T> >
using Array_column_major = Array < T, N,
   ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, IndexMapper_contiguous_column_major<N>, Allocator>>
>;

template <class T, int N, class Config>
class ArrayRef : public Array<T, N, Config>
{
public:
   using array_type = Array<T, N, Config>;

   ArrayRef(array_type& array, const index_type& origin, const index_type& shape, const index_type& stride) :
      array_type(array, origin, shape, stride)
   {}

   ArrayRef& operator=(const array_type& array)
   {
      ensure(array.shape() == this->shape(), "must have the same shape!");
      return *this;
   }

   ArrayRef& operator=(const array_type_ref& array)
   {
      ensure(array.shape() == this->shape(), "must have the same shape!");
      return *this;
   }
};

DECLARE_NAMESPACE_END
