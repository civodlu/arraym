#pragma once

DECLARE_NAMESPACE_NLL

template <class T, size_t N, class Config>
class ArrayRef;

template <class A>
class Expr;

template <class Array>
class ArrayProcessor_contiguous_byDimension;

namespace details
{
   template <class T, class T2, size_t N, class Config, class Config2, class Op>
   void _iterate_array_constarray(Array<T, N, Config>& a1, const Array<T2, N, Config2>& a2, Op& op);

   template <class Array>
   class ArrayProcessor_contiguous_base;
}

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

 Internally, the arithmetic operations used on arrays are controlled by 3 template: @ref array_use_naive,
 @ref array_use_blas, @ref array_use_vectorization. For a given array, only one of these template must
 have a true value. Implemented with a layered approach:
 - basic linear algebra building blocks for arrays such as @ref details::array_add, have a separate implementation
   for each type (naive, BLAS, vectorized)
 - typical operators are defined. If BLAS is enabled and template expression enabled, template based expression operators
   are selected, if not, simple operators.
 */
template <class T, size_t N, class ConfigT = ArrayTraitsConfig<T, N>>
class Array : public ArrayTraits<Array<T, N, ConfigT>, ConfigT>
{
public:
   using Config         = ConfigT;
   using Memory         = typename Config::Memory;
   using allocator_type = typename Config::allocator_type;
   using ConstArray     = Array<const T, N, typename Config::template rebind<const T>::other>;

   using value_type           = T;
   using array_type           = Array<T, N, Config>;
   using array_type_ref       = ArrayRef<T, N, Config>;
   using const_array_type_ref = typename ConstArray::array_type_ref;
   using traits_type          = ArrayTraits<Array<T, N, ConfigT>, ConfigT>;
   using pointer_type         = typename Memory::pointer_type;
   using const_pointer_type   = typename Memory::const_pointer_type;
   using reference_type       = T&;
   using const_reference_type = const T&;
   using index_type           = StaticVector<ui32, N>;
   using diterator            = typename Memory::diterator;
   using const_diterator      = typename Memory::const_diterator;

   static const size_t RANK = N;

   template <class T2>
   struct rebind
   {
      using other = Array<T2, N, typename ConfigT::template rebind<T2>::other>;
   };

   template <size_t N2>
   struct rebind_dim
   {
      using other = Array<T, N2, typename ConfigT::template rebind_dim<N2>::other>;
   };

   template <class... Values>
   struct is_unpacked_arguments
   {
      static const bool value = sizeof...(Values) == RANK && is_same<Values...>::value && std::is_integral<typename first<Values...>::type>::value &&
                                !std::is_same<array_type, typename remove_cvr<typename first<Values...>::type>::type>::value;
   };

   // is this an example of: https://connect.microsoft.com/VisualStudio/feedback/details/1571800/false-positive-warning-c4520-multiple-default-constructors-specified
   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   Array(Values&&... values) : _memory(index_type{values...})
   {
   }

   Array(const index_type& shape, T default_value = T(), const allocator_type& allocator = allocator_type()) : _memory(shape, default_value, allocator)
   {
   }

   /**
   @brief create a shared sub-block
   */
   Array(Array& array, const index_type& origin, const index_type& shape, const index_type& stride) : _memory(array._memory, origin, shape, stride)
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(origin[n] < array.shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(origin[n] + shape[n] <= array.shape()[n], "out of bounds!");
      }
#endif
   }

   template <class T2, class Config2>
   explicit Array( const Array<T2, N, Config2>& array ) : Array( array.shape() )
   {
      // TODO: move to memory
      auto op = [&]( T* a1_pointer, ui32 a1_stride, const T2* a2_pointer, ui32 a2_stride, ui32 nb_elements )
      {
         details::static_cast_naive( a1_pointer, a1_stride, a2_pointer, a2_stride, nb_elements );
      };

      _iterate_array_constarray( *this, array, op );
   }

   Array(const allocator_type& allocator = allocator_type()) : _memory(allocator)
   {
   }

   Array(const Memory& memory) : _memory(memory)
   {
   }

   Array(Memory&& memory) : _memory(std::forward<Memory>(memory))
   {
   }

   Array& operator=(const Array& other)
   {
      _copy(other);
      return *this;
   }

   Array(const Array& other)
   {
      _copy(other);
   }

   template <class Config2>
   Array(const Array<value_type, N, Config2>& other)
   {
      _memory = Memory(other._memory);
   }

   Array& operator=(Array&& other)
   {
      _move(std::forward<Array>(other));
      return *this;
   }

   Array(Array&& other)
   {
      _move(std::forward<Array>(other));
   }

   template <class A>
   Array(const Expr<A>& expr)
   {
      *this = expr();
   }

   template <class A>
   Array& operator=(const Expr<A>& expr)
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

      auto getIndexes_matrix = [](const Array&)
      {
         index_type index;
         for (ui32 n = 0; n < RANK; ++n)
         {
            index[n] = RANK - n - 1;
         }
         return index;
      };

      // special case for matrices: the storage is always transposed
      auto index_fun = is_matrix<Array>::value ? getIndexes_matrix : ArrayProcessor_contiguous_byDimension<Array>::getIndexes;
      details::ArrayProcessor_contiguous_base<Array> iterator(*this, index_fun);

      while (hasMoreElements)
      {
         pointer_type ptr_array(nullptr);
         hasMoreElements        = iterator.accessSingleElement(ptr_array);
         *ptr_array             = *(ptr_initializer++);
      }

      return *this;
   }

   const index_type& shape() const
   {
      return _memory.shape();
   }

   /**
   @brief Rank of the array
   */
   static size_t rank()
   {
      return N;
   }

   void write(std::ostream& f) const
   {
      ensure(0, "@TODO implement");
   }

   void read(std::istream& f)
   {
      ensure(0, "@TODO implement");
   }

   size_t size() const
   {
      size_t s = 1;
      for (int n = 0; n < N; ++n)
      {
         s *= _memory.shape()[n];
      }
      return s;
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   reference_type operator()(const Values&... values)
   {
      index_type index = {values...};
      return operator()(index);
   }

   template <typename... Values, typename = typename std::enable_if<is_unpacked_arguments<Values...>::value>::type>
   const_reference_type operator()(const Values&... values) const
   {
      index_type index = {values...};
      return operator()(index);
   }

   reference_type operator()(const index_type& index)
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(index[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return *_memory.at(index);
   }

   const_reference_type operator()(const index_type& index) const
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
      return subarray(min_index_inclusive, max_index_inclusive);
   }

   array_type_ref subarray(const index_type& min_index_inclusive, const index_type& max_index_inclusive)
   {
#ifndef NDEBUG
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(min_index_inclusive[n] < this->shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(max_index_inclusive[n] < this->shape()[n], "out of bounds!");
         NLL_FAST_ASSERT(min_index_inclusive[n] <= max_index_inclusive[n], "min > max!");
      }
#endif
      const auto size = max_index_inclusive - min_index_inclusive + 1;
      return array_type_ref(*this, min_index_inclusive, size, index_type(1));
   }

   ConstArray asConst() const
   {
      auto m = this->getMemory().asConst();
      auto array    = ConstArray(std::move(m));
      return array;
   }

   const_array_type_ref subarray(const index_type& min_index_inclusive, const index_type& max_index_inclusive) const
   {
      // here we create a new array type which embed the const in the data type so that we really can't modify the array
      return asConst().subarray(min_index_inclusive, max_index_inclusive);
   }
   
   const_array_type_ref operator()(const index_type& min_index_inclusive, const index_type& max_index_inclusive) const
   {
      return subarray(min_index_inclusive, max_index_inclusive);
   }

   /**
    @brief If the argument is a list of @ref RangeA the value is true, false otherwise
    */
   template <typename... Args>
   struct is_range_list
   {
      static const bool value = is_same_nocvr<R, Args...>::value && sizeof...(Args) == N;
   };

   /**
    @tparam Args a list of range[N](min_inclusive, max_exclusive), one for each dimension of the array
    */
   template <typename... Args, typename = typename std::enable_if<is_range_list<Args...>::value>::type>
   array_type_ref operator()(Args&&...args)
   {
      using range_type = typename std::remove_reference<typename first<Args...>::type>::type;
      const range_type ranges[ N ] = { args... };
      index_type min_index_inclusive;
      index_type max_index_inclusive;

      for (size_t n = 0; n < N; ++n)
      {
         auto min_value = *ranges[n].begin();
         auto max_value = *ranges[n].end();
         if (ranges[n] != rangeAll)
         {
            if (min_value < 0)
            {
               min_value = shape()[n] + min_value;
            }
            if (max_value < 0)
            {
               max_value = shape()[n] + max_value + 1;
            }
         }
         else
         {
            min_value = 0;
            max_value = shape()[n];
         }
         NLL_FAST_ASSERT(min_value <= max_value, "min > max");
         min_index_inclusive[n] = min_value;
         max_index_inclusive[n] = max_value - 1;
      }

      return subarray(min_index_inclusive, max_index_inclusive);
   }

   template <typename... Args, typename = typename std::enable_if<is_range_list<Args...>::value>::type>
   const_array_type_ref operator()(Args&&...args) const
   {
      return asConst()(std::forward<Args>(args)...);
   }

   diterator beginDim(ui32 dim, const index_type& indexN)
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.beginDim(dim, indexN);
   }

   const_diterator beginDim(ui32 dim, const index_type& indexN) const
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.beginDim(dim, indexN);
   }

   diterator endDim(ui32 dim, const index_type& indexN)
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.endDim(dim, indexN);
   }

   const_diterator endDim(ui32 dim, const index_type& indexN) const
   {
#ifndef NDEBUG
      NLL_FAST_ASSERT(dim < N, "out of bound!");
      for (int n = 0; n < N; ++n)
      {
         NLL_FAST_ASSERT(indexN[n] < this->shape()[n], "out of bounds!");
      }
#endif
      return _memory.endDim(dim, indexN);
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
      for (auto s : shape())
      {
         if (s == 0)
         {
            return true;
         }
      }
      return false;
   }

   template <class T2>
   typename rebind<T2>::other staticCastTo() const
   {
      using other = typename rebind<T2>::other;
      return other( *this );
   }

   template <size_t slicing_dimension>
   using SlicingMemory = typename Memory::template slice_type<slicing_dimension>::type;

   template <size_t slicing_dimension>
   using SlicingArray = Array<T, N - 1, ArrayTraitsConfig<T, N - 1, allocator_type, SlicingMemory<slicing_dimension>>>;

   template <size_t slicing_dimension>
   SlicingArray<slicing_dimension> slice(const index_type& index) const
   {
      return _memory.slice<slicing_dimension>(index);
   }

private:
   void _move(array_type&& src)
   {
      if (this != &src)
      {
         static_cast<traits_type&>(*this) = std::move(src);
         _memory                          = std::move(src._memory);
      }
   }

   void _copy(const array_type& src)
   {
      static_cast<traits_type&>(*this) = src; // make sure th base class is copied
      _memory                          = src._memory;
   }

public:
   // TODO friend template class
//protected:
   Memory _memory;
};

/**
@brief Default matrix type, following Fortran column-major style
*/
template <class T, class Mapper = IndexMapper_contiguous_matrix_column_major, class Allocator = std::allocator<T>>
using Matrix = Array<T, 2, ArrayTraitsConfig<T, 2, Allocator, Memory_contiguous<T, 2, Mapper, Allocator>>>;

template <class T, class Allocator = std::allocator<T>>
using Matrix_row_major = Matrix<T, IndexMapper_contiguous_matrix_row_major, Allocator>;

template <class T, class Allocator = std::allocator<T>>
using Matrix_column_major = Matrix<T, IndexMapper_contiguous_matrix_column_major, Allocator>;

template <class T, size_t N, class Allocator = std::allocator<T>>
using Array_row_major = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, IndexMapper_contiguous_row_major<N>, Allocator>>>;

template <class T, size_t N, class Allocator = std::allocator<T>>
using Array_row_major_multislice = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_multislice<T, N, IndexMapper_multislice<N, N - 1>, Allocator>>>;

template <class T, size_t N, class Allocator = std::allocator<T>>
using Array_column_major = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, IndexMapper_contiguous_column_major<N>, Allocator>>>;

#ifdef WITH_CUDA
template <class T, size_t N, class Allocator = AllocatorCuda<T>>
using Array_cuda_column_major = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_cuda_contiguous_column_major<T, N, Allocator>>>;
#endif

template <class T, class Allocator = std::allocator<T>>
using Vector = Array_row_major<T, 1, Allocator>;

/**
 @brief Matrix row major with a default small stack based memory allocated

 The goal is to improve memory locality
 */
template <class T, size_t stack_size>
using MatrixSmall_row_major = Matrix_row_major<T, AllocatorSingleStaticMemory<T, stack_size>>;

/**
@brief Matrix row major with a default small stack based memory allocated

The goal is to improve memory locality
*/
template <class T, size_t stack_size>
using MatrixSmall_column_major = Matrix_column_major<T, AllocatorSingleStaticMemory<T, stack_size>>;

/**
@brief Vector with a default small stack based memory allocated

The goal is to improve memory locality
*/
template <class T, size_t stack_size>
using VectorSmall = Array_row_major<T, 1, AllocatorSingleStaticMemory<T, stack_size>>;

/**
 @brief ArrayRef has a different semantic (reference based) from array (i.e., value based)

 When an ArrayRef is modified, it modifies the referenced array instead of creating a copy
 */
template <class T, size_t N, class Config>
class ArrayRef : public Array<T, N, Config>
{
public:
   using array_type = Array<T, N, Config>;
   using index_type = typename array_type::index_type;

   /**
    @brief Construct an array ref from an array
    */
   explicit ArrayRef(array_type& array) : array_type(array, index_type(), array.shape(), index_type(1))
   {
   }

   /**
   @brief Construct an array ref from a sub-array
   */
   ArrayRef(array_type& array, const index_type& origin, const index_type& shape, const index_type& stride) : array_type(array, origin, shape, stride)
   {
   }

   ArrayRef& operator=(const array_type& array)
   {
      ensure(array.shape() == this->shape(), "must have the same shape!");
      auto op = [&](T* y_pointer, ui32 y_stride, const T* x_pointer, ui32 x_stride, ui32 nb_elements) {
         details::copy_naive(y_pointer, y_stride, x_pointer, x_stride, nb_elements);
      };
      iterate_array_constarray(*this, array, op);
      return *this;
   }

   ArrayRef& operator=(T value)
   {
      auto op = [&](T* y_pointer, ui32 y_stride, ui32 nb_elements) { details::set_naive(y_pointer, y_stride, nb_elements, value); };
      iterate_array(*this, op);
      return *this;
   }
};

/**
@brief Specifies if for a specific type of array, we can use BLAS functions
In particular, float and double types can be used with BLAS.

For any array, only one of @ref array_use_vectorization, @ref array_use_naive, @ref array_use_blas must be true
*/
template <class Array>
struct array_use_blas : public std::false_type
{
};

#ifdef WITH_OPENBLAS
template <size_t N, class Config>
struct array_use_blas<Array<float, N, Config>> : public std::true_type
{
};

template <size_t N, class Config>
struct array_use_blas<Array<double, N, Config>> : public std::true_type
{
};
#endif

/**
@todo not supported yet

For any array, only one of @ref array_use_vectorization, @ref array_use_naive, @ref array_use_blas must be true
*/
template <class Array>
struct array_use_vectorization : public std::false_type
{
};

/**
@brief if values is true, it means we need to use the naive set of operations for this array

For any array, only one of @ref array_use_vectorization, @ref array_use_naive, @ref array_use_blas must be true
*/
template <class Array>
struct array_use_naive
{
   static const bool value = !array_use_vectorization<Array>::value && !array_use_blas<Array>::value;
};

#ifdef WITH_EXPRESSION_TEMPLATE
template <class Array>
struct array_use_naive_operator : public std::false_type
{
};
#else
template <class Array>
struct array_use_naive_operator : public std::true_type
{
};
#endif


namespace details
{
   template <class T, size_t N, class ConfigT>
   StaticVector<ui32, N> getFastestVaryingIndexes( const Array<T, N, ConfigT>& array );

   template <class Memory>
   StaticVector<ui32, Memory::RANK> getFastestVaryingIndexesMemory( const Memory& memory );
}

/**
@brief returns true if two array have similar data ordering. (i.e., using an iterator, we point to the same
index for both arrays)
@todo needs to be extensible (using class specialization) for custom types!
@todo this test is not 100% accurate: ambiguous in case the one dim == 1
*/
template <class T, size_t N, class Config, class Config2>
bool same_data_ordering(const Array<T, N, Config>& a1, const Array<T, N, Config2>& a2)
{
   // manually resolve the ambiguity, still problems for higher dimension
   if (N == 2 && (a1.shape()[0] == 1 || a1.shape()[1] == 1))
   {
      return true;
   }
   const auto i1 = details::getFastestVaryingIndexes<T, N, Config>(a1);
   const auto i2 = details::getFastestVaryingIndexes<T, N, Config2>( a2 );
   return i1 == i2;
}

template <class Memory1, class Memory2>
bool same_data_ordering_memory(const Memory1& a1, const Memory2& a2)
{
   const auto i1 = details::getFastestVaryingIndexesMemory(a1);
   const auto i2 = details::getFastestVaryingIndexesMemory(a2);
   return i1 == i2;
}

/**
@brief Returns true if an array is based on a single slice of contiguous memory
@note this doesn't mean there is not gap between dimensions (e.g., we have a sub-array)
*/
template <class Array>
struct IsArrayLayoutContiguous
{
   static const bool value = std::is_base_of<memory_layout_contiguous, typename Array::Memory>::value;
};

/**
@brief Returns true if an array is based on a single slice or multiple slices of contiguous memory
@note this doesn't mean there is not gap between dimensions (e.g., we have a sub-array)
*/
template <class Array>
struct IsArrayLayoutLinear
{
   static const bool value = std::is_base_of<memory_layout_linear, typename Array::Memory>::value;
};

/**
 @brief Returns true if the array is fully contiguous, meaning that the array occupies a single block of contiguous memory
        with no gap between elements (i.e., can't generally be a sub-array)
 */
template <class T, size_t N, class Config>
bool is_array_fully_contiguous(const Array<T, N, Config>& a1)
{
   return is_memory_fully_contiguous<typename Array<T, N, Config>::Memory>(a1.getMemory());
}

namespace details
{
   template <class Array>
   struct GetBaseMemory
   {
      using pointer_type = typename Array::pointer_type;

      pointer_type operator()(const Array& array)
      {
         ensure(0, "<Array> doesn't define a base memory! Invalid call");
      }
   };

   template <class T, size_t N, class Allocator, class Mapper, class PointerType>
   struct GetBaseMemory<Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, Mapper, Allocator, PointerType>>>>
   {
      using array_type = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous<T, N, Mapper, Allocator, PointerType>>>;
      using pointer_type = typename array_type::pointer_type;

      pointer_type operator()(const array_type& array)
      {
         return array.getMemory()._getBaseData();
      }
   };

   //
   // TODO extend your custom types here
   //
}

/**
 @brief Return the base memory of an array

 This method is only available for specific memory type (e.g., contiguous), meaning that 
 the data can be stored in a single contiguous memory store
 */
template <class Array>
typename Array::pointer_type array_base_memory(const Array& array)
{
   return details::GetBaseMemory<Array>()(array);
}

DECLARE_NAMESPACE_NLL_END
