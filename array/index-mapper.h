#pragma once

DECLARE_NAMESPACE_NLL

namespace details
{
/// helper to compute the strides for index mapper of a contiguous memory block, row major
template <size_t N>
struct Mapper_stride_row_major
{
public:
   using index_type = StaticVector<ui32, N>;

   index_type operator()(const index_type& shape) const
   {
      index_type strides;
      strides[0] = 1;
      extend_stride(strides, shape, 1, 0);
      return strides;
   }

   /**
    @brief Extend an existing physical striding to more dimensions
    @param index_start first first index to fill. Assuming [physicalStrides:index_start-1] are already correctly populated
    @param last_index the last known index which physical stride populated. This will be used as seed to extend the physical strides

    [index_start..N] : these are the dimensions to be extended
    [0..index_start-1] : these are the already mapped dimensions
    */
   void extend_stride(index_type& physicalStrides, const index_type& shape, int index_start, int last_index) const
   {
      for (int n = index_start; n < N; ++n)
      {
         const ui32 stride  = physicalStrides[last_index] * shape[last_index];
         last_index         = n;
         physicalStrides[n] = stride;
      }
   }

   template <size_t dim>
   struct rebind
   {
      using other = Mapper_stride_row_major<dim>;
   };
};

/// helper to compute the strides for index mapper of a multislice memory block
template <size_t N, size_t Z_INDEX_>
struct Mapper_multisplice_stride_row_major
{
public:
   using index_type = StaticVector<ui32, N>;

   index_type operator()(const index_type& shape) const
   {
      index_type strides;
      ui32 stride = 1;
      for (size_t n = 0; n < N; ++n)
      {
         if (n != Z_INDEX_)
         {
            strides[n] = stride;
            stride *= shape[n];
         }
      }
      return strides;
   }

   template <size_t dim, size_t NEW_Z_INDEX>
   struct rebind
   {
      using other = Mapper_multisplice_stride_row_major<dim, NEW_Z_INDEX>;
   };
};

/// helper to compute the strides for index mapper of a contiguous memory block, column major
template <size_t N>
struct Mapper_stride_column_major
{
public:
   using index_type = StaticVector<ui32, N>;

   index_type operator()(const index_type& shape) const
   {
      index_type strides;

      ui32 stride = 1;
      for (int n = N - 1; n >= 0; --n)
      {
         strides[n] = stride;
         stride *= shape[n];
      }
      return strides;
   }

   /**
    @brief Extend an existing physical striding to more dimensions
    @param index_start first first index to fill. Assuming [physicalStrides:index_start-1] are already correctly populated
    @param last_index the last known index which physical stride populated. This will be used as seed to extend the physical strides

    [index_start..N] : these are the dimensions to be extended
    [0..index_start-1] : these are the already mapped dimensions
    */
   void extend_stride(index_type& physicalStrides, const index_type& shape, int index_start, int last_index) const
   {
      for (int n = N - 1; n >= index_start; --n)
      {
         const ui32 stride  = shape[n] * physicalStrides[last_index];
         physicalStrides[n] = stride;
         last_index         = n;
      }
   }

   template <size_t dim>
   struct rebind
   {
      using other = Mapper_stride_column_major<dim>;
   };
};
}

/**
@brief memory will be stored linearly (can be expressed as origin + x * strides within slices)
*/
struct memory_layout_linear
{
};

/**
@brief memory will be stored contiguously
*/
struct memory_layout_contiguous : public memory_layout_linear
{
};

/**
@brief memory will be stored contiguously EXCEPT in the last dimensions.

E.g., for Dim=3, (x, y, ...) are stored in contiguous portions of memory,
but not (..., ..., n) and (..., ..., n+1)
*/
struct memory_layout_multislice_z : public memory_layout_linear
{
};

/**
@brief index mapper for multislice z memory layout.
*/
template <size_t N, class Mapper = details::Mapper_stride_row_major<N>>
class IndexMapper_contiguous : public memory_layout_contiguous
{
public:
   using index_type  = StaticVector<ui32, N>;
   using IndexMapper = IndexMapper_contiguous<N, Mapper>;
   using mapper_type = Mapper;

   template <size_t dim>
   struct rebind
   {
      using other = IndexMapper_contiguous<dim, typename Mapper::template rebind<dim>::other>;
   };

   /**
   @param shape the size of the area to map
   */
   void init(ui32 origin, const index_type& shape)
   {
      index_type strides = Mapper()(shape);

      _origin          = origin;
      _physicalStrides = strides;
   }

   /**
   @param physicalStrides the strides of the area to map, considering the underlying memory (i.e,
   if we reference a sub-block, this must be factored in the stride)
   */
   void init(const index_type& physicalStrides)
   {
      _origin          = 0;
      _physicalStrides = physicalStrides;
   }

   ui32 offset(const index_type& index) const
   {
      return dot(index, _physicalStrides) + _origin;
   }

   IndexMapper submap(const index_type& origin, const index_type& shape, const index_type& strides) const
   {
      const auto newPhysicalStrides = _physicalStrides * strides;
      const auto newOrigin          = offset(origin);

      IndexMapper newMapper;
      newMapper.init(newOrigin, shape);
      newMapper._physicalStrides = newPhysicalStrides;
      return newMapper;
   }

   // this is not part of the generic interface but can be specifically used for a memory/mapper
   const index_type& _getPhysicalStrides() const
   {
      return _physicalStrides;
   }

   // this is not part of the generic interface but can be specifically used for a memory/mapper
   void _setPhysicalStrides(const index_type& physicalStrides)
   {
      _physicalStrides = physicalStrides;
   }

   /**
    @brief slice an array following @p dimension at the position @p index
    */
   template <size_t dimension>
   typename rebind<N - 1>::other slice(const index_type& UNUSED(index)) const
   {
      typename rebind<N - 1>::other sliced_index;
      sliced_index._origin = 0; // the _data will be set to index so start from 0

      size_t current_dim = 0;
      for (size_t n = 0; n < N; ++n)
      {
         if (n != dimension)
         {
            sliced_index._physicalStrides[current_dim++] = _physicalStrides[n];
         }
      }
      return sliced_index;
   }

   //private:
   ui32 _origin;
   index_type _physicalStrides;
};

/**
 @brief Here we create a different mapper so that we can clearly identity if the type is a matrix, with the specific
        matrix semantic
 */
template <size_t N, class Mapper>
struct IndexMapper_contiguous_matrix : public IndexMapper_contiguous<N, Mapper>
{
   using index_type  = StaticVector<ui32, N>;
   using IndexMapper = IndexMapper_contiguous_matrix<N, Mapper>;
   using mapper_type = Mapper;

   template <size_t dim>
   struct rebind
   {
      using other = IndexMapper_contiguous_matrix<dim, typename Mapper::template rebind<dim>::other>;
   };

   IndexMapper_contiguous_matrix(const IndexMapper_contiguous<N, Mapper>& base)
   {
      this->_origin          = base._origin;
      this->_physicalStrides = base._physicalStrides;
   }

   IndexMapper_contiguous_matrix()
   {
   }
};

template <size_t N>
using IndexMapper_contiguous_row_major = IndexMapper_contiguous<N, details::Mapper_stride_row_major<N>>;

template <size_t N>
using IndexMapper_contiguous_column_major = IndexMapper_contiguous<N, details::Mapper_stride_column_major<N>>;

/**
@brief index mapper for row major matrices

We have a special mapper for matrix and use this to differentiate between Matrix and
array. Else an array <float, 2> will be considered a matrix
*/
using IndexMapper_contiguous_matrix_row_major = IndexMapper_contiguous_matrix<2, details::Mapper_stride_column_major<2>>;

/**
@brief index mapper for column major matrices

We have a special mapper for matrix and use this to differentiate between Matrix and
array. Else an array <float, 2> will be considered a matrix
*/
using IndexMapper_contiguous_matrix_column_major = IndexMapper_contiguous_matrix<2, details::Mapper_stride_row_major<2>>;

/**
@brief index mapper for multislice z memory layout.
*/
template <size_t N, size_t Z_INDEX_, class Mapper = details::Mapper_multisplice_stride_row_major<N, Z_INDEX_>>
class IndexMapper_multislice : public memory_layout_multislice_z
{
   // if we rebind the Z dimension, we end up with full contiguous memory so default to a contiguous mapper!
   template <size_t dim>
   struct rebind_z
   {
      using other = IndexMapper_contiguous<dim, details::Mapper_stride_row_major<dim>>;
   };

   template <size_t dim, size_t NEW_Z_INDEX>
   struct rebind_notz
   {
      using other = IndexMapper_multislice<dim, NEW_Z_INDEX, typename Mapper::template rebind<dim, NEW_Z_INDEX>::other>;
   };

public:
   using index_type            = StaticVector<ui32, N>;
   static const size_t Z_INDEX = Z_INDEX_;
   using IndexMapper           = IndexMapper_multislice<N, Z_INDEX, Mapper>;
   using mapper_type           = Mapper;

   // if NEW_Z_INDEX < 0, this means we have destroyed the Z_INDEX
   template <size_t dim, size_t NEW_Z_INDEX>
   using rebind = typename std::conditional < NEW_Z_INDEX<0, rebind_z<dim>, rebind_notz<dim, NEW_Z_INDEX>>::type;

   /**
   @param shape the size of the area to map
   */
   void init(ui32 origin, const index_type& shape)
   {
      index_type strides = Mapper()(shape);

      _origin                   = origin;
      _physicalStrides          = strides;
      _physicalStrides[Z_INDEX] = 0; // in slice offset should be independent of Z_INDEX axis
   }

   void init(const index_type& physicalStrides)
   {
      _origin                   = 0; // the slices' ptr should already be set to origin
      _physicalStrides          = physicalStrides;
      _physicalStrides[Z_INDEX] = 0; // in slice offset should be independent of Z_INDEX axis
   }

   ui32 offset(const index_type& index) const
   {
      return dot(index, _physicalStrides) + _origin;
   }

   IndexMapper submap(const index_type& origin, const index_type& shape, const index_type& strides) const
   {
      const auto newPhysicalStrides = _physicalStrides * strides;
      const auto newOrigin          = offset(origin);

      IndexMapper newMapper;
      newMapper.init(newOrigin, shape);
      newMapper._physicalStrides = newPhysicalStrides;
      return newMapper;
   }

public:
   // this is not part of the generic interface but can be specifically used for a memory/mapper
   const index_type& _getPhysicalStrides() const
   {
      return _physicalStrides;
   }

private:
   ui32 _origin;
   index_type _physicalStrides;
};

DECLARE_NAMESPACE_NLL_END
