#pragma once

DECLARE_NAMESPACE_NLL

template <class T, int N, class Config>
class ArrayRef;

template <class A>
class Expr;

/**
 @brief Represents a multi-dimensional array with value based semantic

 @tparam T the type to be stored by the array
 @tparam N the number of dimensions for the array
 @tparam ConfigT a configuration parameter, specifying e.g., how the array's memory is stored internally (e.g., @ref ArrayTraitsConfig)

 The array can simply be accessed using subscripting or indices:
 @code
 Array<float, 2> array(2, 3);
 array(0, 0) = 1;
 array({1, 1}) = 2;
 @endcode

 The array has a value based semantics, meaning that we prefer copying the value of the array
 rather than sharing the memory between arrays.
 
 @code
 Array<float, 2> array1(2, 3);
 Array<float, 2> array2(1, 1)
 array2 = array1; // the array2 current memory is released and the content of array1 is copied
 @endcode

 Although for performance and convenience it is useful to reference a portion of the array and modify
 it as required (@ref ArrayRef). This can be done using a min and max (inclusive) index:
 @code
 Array<float, 2> array(10, 10);
 auto sub_array = array({2, 2}, {5, 5});  // this points to min=(2, 2) to max=(5, 5) of array
 sub_array = 42;  // the referenced array will be updated
 @endcode

 while a @ref ArrayRef is in use, the original @ref Array must be kept alive.
 */
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
   using const_reference_type = const T&;
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

   template <class A>
   Array( const Expr<A>& expr )
   {
      *this = expr();
   }

   template <class A>
   Array& operator=( const Expr<A>& expr )
   {
      *this = expr();
      return *this;
   }

   /**
    @brief The array will be filled with a list of values. The array will be filled by dimension order (first dim[0], then dim[1] and so on)
    
    It is not the fastest filling order for e.g., column major memory, but typically this is just a convenience function.
    */
   Array& operator=(const std::initializer_list<T>& list)
   {
      ensure(list.size() == this->size(), "initializer and current array must have the same size!");
      auto ptr_initializer = list.begin();

      bool hasMoreElements = true;
      ArrayProcessor_contiguous_byDimension<Array> iterator(*this);
      while (hasMoreElements)
      {
         pointer_type ptr_array = 0;
         hasMoreElements = iterator.accessSingleElement(ptr_array);
         *ptr_array = *(ptr_initializer++);
      }

      return *this;
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

   size_t size() const
   {
      size_t s = 1;
      for ( int n = 0; n < N; ++n )
      {
         s *= _memory.getShape()[ n ];
      }
      return s;
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   reference_type operator()( const Values&... values )
   {
      index_type index = { values... };
      return operator()( index );
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   const_reference_type operator()( const Values&... values ) const
   {
      index_type index = { values... };
      return operator()( index );
   }

   reference_type operator()( const index_type& index )
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(index[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return *_memory.at( index );
   }

   const_reference_type operator()( const index_type& index ) const
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(index[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return *_memory.at( index );
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

   Memory& getMemory()
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

/**
 @brief ArrayRef has a different semantic (reference based) from array (i.e., value based)

 When an ArrayRef is modified, it modifies the referenced array instead of creating a copy
 */
template <class T, int N, class Config>
class ArrayRef : public Array<T, N, Config>
{
public:
   using array_type = Array<T, N, Config>;

   /**
    @brief Construct an array ref from an array
    */
   explicit ArrayRef(Array& array) : array_type(array, index_type(), array.shape(), index_type(1))
   {}

   /**
   @brief Construct an array ref from a sub-array
   */
   ArrayRef(array_type& array, const index_type& origin, const index_type& shape, const index_type& stride) :
      array_type(array, origin, shape, stride)
   {}

   ArrayRef& operator=(const array_type& array)
   {
      ensure(array.shape() == this->shape(), "must have the same shape!");
      return *this;
   }
};

DECLARE_NAMESPACE_END
