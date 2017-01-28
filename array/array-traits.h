#pragma once

DECLARE_NAMESPACE_NLL
template <class T, size_t N, class ConfigT>
class Array;

template <class T, size_t N, class Allocator = std::allocator<T>, class MemoryT = Memory_contiguous_row_major<T, N, Allocator>>
struct ArrayTraitsConfig
{
   static const size_t RANK = N;
   using Memory             = MemoryT;
   using allocator_type     = Allocator;
   using value_type         = T;

   /**
   @brief Rebind the trait to a different type
   */
   template <class T2>
   struct rebind
   {
      using Memory = typename MemoryT::template rebind<T2>::other;

      using other = ArrayTraitsConfig<T2, N, typename Memory::allocator_type, Memory>;
   };

   /**
   @brief Rebind the trait to a different type
   */
   template <size_t N2>
   struct rebind_dim
   {
      using Memory = typename MemoryT::template rebind_dim<N2>::other;
      using other = ArrayTraitsConfig<T, N2, typename Memory::allocator_type, Memory>;
   };
};

/**
We need @p Config for VS2013. Type is not fully constructed when ArrayTraits is instantiated
*/
template <class Array, class Config, class Enable = void>
class ArrayTraits;

template <class Array, class Config>
class ArrayTraitsDefault
{
public:
   template <class functor>
   void fill(functor f)
   {
      auto& array = static_cast<Array&>(*this);
      //detail::generic_fill( array, f );
      ensure(0, "@TODO implement");
   }
};

template <class Array>
class is_matrix : public std::false_type
{
};

template <class T, class Allocator>
class is_matrix<Array<T, 2, ArrayTraitsConfig<T, 2, Allocator, Memory_contiguous<T, 2, IndexMapper_contiguous_matrix_column_major, Allocator>>>> : public std::true_type
{
};

template <class T, class Allocator>
class is_matrix<Array<T, 2, ArrayTraitsConfig<T, 2, Allocator, Memory_contiguous<T, 2, IndexMapper_contiguous_matrix_row_major, Allocator>>>> : public std::true_type
{
};

#ifdef WITH_CUDA
template <class T, class Allocator>
class is_matrix<Array<T, 2, ArrayTraitsConfig<T, 2, Allocator, Memory_cuda_contiguous_column_major<T, 2, Allocator, IndexMapper_contiguous_matrix_column_major>>>> : public std::true_type
{
};
#endif

// any other type: no particular specialization
template <class ArrayT, class Config>
class ArrayTraits<ArrayT, Config, typename std::enable_if<Config::RANK != 2>::type> : public ArrayTraitsDefault<ArrayT, Config>
{
};

// any other type: no particular specialization
template <class ArrayT, class Config>
class ArrayTraits<ArrayT, Config, typename std::enable_if<Config::RANK == 2 && (!is_matrix<ArrayT>::value)>::type> : public ArrayTraitsDefault<ArrayT, Config>
{
};

// matrix type: create additional API
template <class Array, class Config>
class ArrayTraits<Array, Config, typename std::enable_if<Config::RANK == 2 && (is_matrix<Array>::value)>::type> : public ArrayTraitsDefault<Array, Config>
{
public:
   size_t rows() const
   {
      auto& array = static_cast<const Array&>(*this);
      return array.shape()[0];
   }

   size_t columns() const
   {
      auto& array = static_cast<const Array&>(*this);
      return array.shape()[1];
   }

   size_t sizex() const
   {
      return columns();
   }

   size_t sizey() const
   {
      return rows();
   }
};
DECLARE_NAMESPACE_NLL_END