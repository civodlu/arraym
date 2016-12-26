#pragma once

/**
 @file

 Utility functions to convert array to vector and vice versa

 @TODO here we are losing constness. Can we avoid this using, e.g., array.asConst() without complexifying the API?
 */
DECLARE_NAMESPACE_NLL

/**
 @brief View the 1D vector as a matrix row major. The data is shared between the two
 */
template <class T, class Allocator>
Matrix_row_major<T, Allocator> as_matrix_row_major(const Vector<T, Allocator>& v, const typename Matrix_row_major<T, Allocator>::index_type& shape)
{
   using matrix_type = Matrix_row_major<T, Allocator>;

   // base the mapping on the vector's stride, then extend it to higher dimension
   using mapper_type = typename matrix_type::Memory::index_mapper::mapper_type;
   typename matrix_type::index_type physical_stride(v.getMemory().getIndexMapper()._getPhysicalStrides()[0], 0);
   mapper_type().extend_stride(physical_stride, shape, 1, 0);

   matrix_type matrix(typename matrix_type::Memory(shape, const_cast<T*>(&v(0)), physical_stride, v.getMemory().getAllocator()));
   ensure(matrix.size() == v.size(), "size must match!");
   return matrix;
}

/**
@brief View the 1D vector as a matrix column major. The data is shared between the two
*/
template <class T, class Allocator>
Matrix_column_major<T, Allocator> as_matrix_column_major(const Vector<T, Allocator>& v, const typename Matrix_column_major<T, Allocator>::index_type& shape)
{
   using matrix_type = Matrix_column_major<T, Allocator>;

   // base the mapping on the vector's stride, then extend it to higher dimension
   using mapper_type = typename matrix_type::Memory::index_mapper::mapper_type;
   typename matrix_type::index_type physical_stride(v.getMemory().getIndexMapper()._getPhysicalStrides()[0], 0);
   mapper_type().extend_stride(physical_stride, shape, 1, 0);

   matrix_type matrix(typename matrix_type::Memory(shape, const_cast<T*>(&v(0)), physical_stride, v.getMemory().getAllocator()));
   ensure(matrix.size() == v.size(), "size must match!");
   return matrix;
}

/**
@brief View the N-array as a 1D vector. The data is shared between the two
*/
template <class T, size_t N, class Config>
Vector<T, typename Config::allocator_type> as_vector(const Array<T, N, Config>& a)
{
   // TODO it is only possible if all strides are equal and
   // there is no gap between dimensions. For now, don't do it as rare usecase
   ensure(is_array_fully_contiguous(a), "can't fit a strided array into a single vector with different stride or gap between dimensions");

   using vector_type = Vector<T, typename Config::allocator_type>;
   using array_index_type = typename Array<T, N, Config>::index_type;
   using vector_index_type = typename vector_type::index_type;
   vector_type v(typename vector_type::Memory(
      vector_index_type(a.size()),
      const_cast<T*>(&a(array_index_type())),
      a.getMemory().getAllocator()));
   return v;
}

/**
@brief View the 1D vector as a N-array. The data is shared between the two
*/
template <class T, size_t N, class Allocator>
Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous_row_major<T, N, Allocator>>> as_array_row_major( const Vector<T, Allocator>& v, const StaticVector<ui32, N>& shape )
{
   using array_type = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous_row_major<T, N, Allocator>>>;
   ensure( is_array_fully_contiguous( v ), "can't fit a strided array into a single vector with different stride or gap between dimensions" );

   // base the mapping on the vector's stride, then extend it to higher dimension
   //using mapper_type = typename array_type::Memory::index_mapper::mapper_type;
   //typename array_type::index_type physical_stride;
   //physical_stride[ 0] = v.getMemory().getIndexMapper()._getPhysicalStrides()[ 0 ];
   //mapper_type().extend_stride( physical_stride, shape, 1, 0 );
   //array_type array( typename array_type::Memory( shape, const_cast<T*>( &v( 0 ) ), physical_stride, v.getMemory().getAllocator() ) );

   array_type array( typename array_type::Memory( shape, const_cast<T*>( &v( 0 ) ), v.getMemory().getAllocator() ) );
   ensure(array.size() == v.size(), "must have the same size!");
   return array;
}

/**
@brief View the 1D vector as a N-array. The data is shared between the two
*/
template <class T, size_t N, class Allocator>
Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous_column_major<T, N, Allocator>>> as_array_column_major(const Vector<T, Allocator>& v, const StaticVector<ui32, N>& shape)
{
   using array_type = Array<T, N, ArrayTraitsConfig<T, N, Allocator, Memory_contiguous_column_major<T, N, Allocator>>>;
   ensure( is_array_fully_contiguous( v ), "can't fit a strided array into a single vector with different stride or gap between dimensions" );

   // base the mapping on the vector's stride, then extend it to higher dimension
   using mapper_type = typename array_type::Memory::index_mapper::mapper_type;
   //typename array_type::index_type physical_stride;
   //physical_stride[ 0 ] = v.getMemory().getIndexMapper()._getPhysicalStrides()[ 0 ];
   //mapper_type().extend_stride( physical_stride, shape, 1, 0 );
   //array_type array( typename array_type::Memory( shape, const_cast<T*>( &v( 0 ) ), physical_stride, v.getMemory().getAllocator() ) );
   
   array_type array( typename array_type::Memory( shape, const_cast<T*>( &v( 0 ) ), v.getMemory().getAllocator() ) );
   ensure(array.size() == v.size(), "must have the same size!");
   return array;
}

template <class T, size_t N, class Config, size_t N2>
Array<T, N2, typename Config::template rebind_dim<N2>::other> as_array(const Array<T, N, Config>& v, const StaticVector<ui32, N2>& shape)
{
   using array_type = Array<T, N, Config>;
   static_assert( IsArrayLayoutLinear<array_type>::value, "the array must have a linear layout!" );
   static_assert(N2 >= N, "N2 must be higher!");
   for (size_t n = 0; n < N; ++n)
   {
      ensure(v.shape()[n] == shape[n], "can't change the shape of the original array");
   }

   using other_array_type = Array<T, N2, typename Config::template rebind_dim<N2>::other>;
   using other_mapper_type = typename other_array_type::Memory::index_mapper::mapper_type;
   
   // extend the original stride
   typename other_array_type::index_type physical_stride;
   const auto& original_physical_stride = v.getMemory().getIndexMapper()._getPhysicalStrides();
   for ( size_t n = 0; n < N; ++n )
   {
      physical_stride[ n ] = original_physical_stride[ n ];
   }

   // the last index of stride is the element with the highest value
   auto max_index = std::max_element( original_physical_stride.begin(), original_physical_stride.end() );
   const auto last_element = max_index - original_physical_stride.begin();
   other_mapper_type().extend_stride( physical_stride, shape, N, int(last_element) );

   other_array_type array( typename other_array_type::Memory( shape, const_cast<T*>( &v( typename array_type::index_type() ) ), physical_stride, v.getMemory().getAllocator() ) );
   ensure(array.size() == v.size(), "must have the same size!");
   return array;
}

DECLARE_NAMESPACE_NLL_END