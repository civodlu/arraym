#include <array/forward.h>
#include <tester/register.h>

using namespace nll;

using ui32 = NAMESPACE_NLL::ui32;

DECLARE_NAMESPACE_NLL
using vector3ui = StaticVector < ui32, 3 > ;

namespace details
{
   //
   // TODO DO NOT USE RAND() and use thread local generators
   //
   struct UniformDistribution
   {
      template <class T>
      static T generate( T min, T max, std::false_type UNUSED( isIntegral ) )
      {
         NLL_FAST_ASSERT( min <= max, "invalid range!" );
         return static_cast<T>( rand() ) / RAND_MAX * ( max - min ) + min;
      }

      template <class T>
      static T generate( T min, T max, std::true_type UNUSED( isIntegral ) )
      {
         NLL_FAST_ASSERT( min <= max, "invalid range!" );
         const T interval = max - min + 1;
         if ( interval == 0 )
            return min;
         return ( rand() % interval ) + min;
      }
   };
}

/**
@ingroup core
@brief generate a sample of a specific uniform distribution
@param min the min of the distribution, inclusive
@param max the max of the distribution, inclusive
@return a sample of this distribution
*/
template <class T>
T generateUniformDistribution( T min, T max )
{
   static_assert( std::is_arithmetic<T>::value, "must be a numeric type!" );
   return details::UniformDistribution::generate<T>( min, max, std::is_integral<T>() );
}

DECLARE_NAMESPACE_END

struct TestArray
{
   void testVolumeConstruction_slices()
   {
      NAMESPACE_NLL::Memory_multislice<float, 3> memory( NAMESPACE_NLL::vector3ui( 2, 3, 4 ), 5 );
      const auto& slices = memory._getSlices();
      TESTER_ASSERT( slices.size() == 4 );
      TESTER_ASSERT( slices[ 1 ][ 2 ] == 5 );

      auto p = memory.at( { 0, 0, 1 } );
      TESTER_ASSERT( *p == 5 );
   }

   void testVolumeConstruction_slices_ref()
   {
      testVolumeConstruction_ref_impl<NAMESPACE_NLL::Memory_multislice<int, 3>>();
      testVolumeConstruction_ref_impl<NAMESPACE_NLL::Memory_contiguous<int, 3>>();
   }

   template <class Memory>
   void testVolumeConstruction_ref_impl()
   {
      //
      // initial volume
      //
      const auto size = NAMESPACE_NLL::vector3ui( 50, 51, 52 );
      Memory memory( size, 0 );

      int index = 0;
      for ( size_t z = 0; z < size[ 2 ]; ++z )
      {
         for ( size_t y = 0; y < size[ 1 ]; ++y )
         {
            for ( size_t x = 0; x < size[ 0 ]; ++x )
            {
               *memory.at( { x, y, z } ) = index++;
            }
         }
      }

      const size_t in_slice_size = size[ 0 ] * size[ 1 ];

      TESTER_ASSERT( *memory.at( { 0, 0, 0 } ) == 0 );
      TESTER_ASSERT( *memory.at( { 0, 0, 1 } ) == in_slice_size );
      TESTER_ASSERT( *memory.at( { 0, 0, 2 } ) == in_slice_size * 2 );
      TESTER_ASSERT( *memory.at( { 1, 0, 2 } ) == in_slice_size * 2 + 1 );
      TESTER_ASSERT( *memory.at( { 1, 2, 2 } ) == in_slice_size * 2 + 1 + 2 * size[ 0 ] );


      //
      // reference a volume
      //
      const NAMESPACE_NLL::vector3ui origin = { 2, 3, 4 };
      const NAMESPACE_NLL::vector3ui sub_size = { 10, 11, 12 };
      const NAMESPACE_NLL::vector3ui sub_strides = { 2, 3, 2 };
      Memory memory_ref( memory, origin, sub_size, sub_strides );
      {
         auto value = *memory_ref.at( { 0, 0, 0 } );
         auto expected = *memory.at( origin );
         TESTER_ASSERT( value == expected );
      }

      for ( int n = 0; n < 1000; ++n )
      {
         srand( n );
         const NAMESPACE_NLL::vector3ui displacement =
         {
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size[ 0 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size[ 1 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size[ 2 ] - 1 )
         };
         auto value = *memory_ref.at( displacement );
         auto expected = *memory.at( origin + displacement * sub_strides );
         TESTER_ASSERT( value == expected );
      }

      //
      // Reference a referenced volume with stride
      //
      const NAMESPACE_NLL::vector3ui origin2 = { 2, 1, 0 };
      const NAMESPACE_NLL::vector3ui sub_size2 = { 2, 3, 2 };
      const NAMESPACE_NLL::vector3ui sub_strides2 = { 1, 3, 2 };
      Memory memory_ref2( memory_ref, origin2, sub_size2, sub_strides2 );

      for ( int n = 0; n < 1000; ++n )
      {
         srand( n );
         const NAMESPACE_NLL::vector3ui displacement =
         {
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size2[ 0 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size2[ 1 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, sub_size2[ 2 ] - 1 )
         };
         const auto value = *memory_ref2.at( displacement );
         const auto index_original = ( origin2 + (displacement)* sub_strides2 ) * sub_strides + origin;
         const auto expected = *memory.at( index_original );
         TESTER_ASSERT( value == expected );
      }

      //
      // Test copy
      //
      Memory memory_cpy = memory_ref2;
      TESTER_ASSERT( memory_cpy.getShape() == memory_ref2.getShape() );
      for ( size_t n = 0; n < 500; ++n )
      {
         const NAMESPACE_NLL::vector3ui displacement =
         {
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, memory_ref2.getShape()[ 0 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, memory_ref2.getShape()[ 1 ] - 1 ),
            NAMESPACE_NLL::generateUniformDistribution<ui32>( 0, memory_ref2.getShape()[ 2 ] - 1 )
         };
         TESTER_ASSERT( *memory_cpy.at( displacement ) == *memory_ref2.at( displacement ) );
      }
   }

   void testVolumeMove()
   {
      testVolumeMove_impl<NAMESPACE_NLL::Memory_contiguous<int, 3>>();
      testVolumeMove_impl<NAMESPACE_NLL::Memory_multislice<int, 3>>();
   }

   template <class Memory>
   void testVolumeMove_impl()
   {
      Memory test1( { 4, 5, 6 } );
      const auto ptr = test1.at( { 1, 2, 3 } );

      Memory test1_moved = std::forward<Memory>( test1 );
      const auto ptr_moved = test1_moved.at( { 1, 2, 3 } );

      TESTER_ASSERT( ptr_moved == ptr );

      Memory test2_moved;
      test2_moved = std::forward<Memory>( test1_moved );
      const auto ptr_moved2 = test2_moved.at( { 1, 2, 3 } );
      TESTER_ASSERT( ptr_moved2 == ptr );
   }

   void testArray_construction()
   {
      using Array = NAMESPACE_NLL::Array < int, 3 >;

      Array a1( { 1, 2, 3 }, 42 );
   }

   void testMatrix_construction()
   {
      NAMESPACE_NLL::Matrix<float> m( { 2, 3 }, 0 );
      TESTER_ASSERT( m.rows() == 2 );
      TESTER_ASSERT( m.columns() == 3 );

      m( { 1, 2 } ) = 42;
      TESTER_ASSERT( m( 1, 2 ) == 42 );
   }

   void testMatrix_construction2()
   {
      NAMESPACE_NLL::Matrix<float> m( 2, 3 ); // construction with unpacked arugments
      TESTER_ASSERT( m.rows() == 2 );
      TESTER_ASSERT( m.columns() == 3 );
   }

   void testArray_directional_index()
   {
      testArray_directional_index_impl<NAMESPACE_NLL::Memory_contiguous<int, 3>>();
      testArray_directional_index_impl<NAMESPACE_NLL::Memory_multislice<int, 3>>();
   }

   template <class Memory>
   void testArray_directional_index_impl()
   {
      const NAMESPACE_NLL::vector3ui size = { 40, 41, 42 };
      const NAMESPACE_NLL::vector3ui size_m2 = { 5, 6, 7 };
      const NAMESPACE_NLL::vector3ui stride_m2 = { 2, 1, 3 };
      const NAMESPACE_NLL::vector3ui offset_start_m1 = { 3, 1, 2 };

      Memory m1( size );
      Memory m2( m1, offset_start_m1, size_m2, stride_m2 );

      // init
      int index = 0;
      for ( size_t z = 0; z < size[ 2 ]; ++z )
      {
         for ( size_t y = 0; y < size[ 1 ]; ++y )
         {
            for ( size_t x = 0; x < size[ 0 ]; ++x )
            {
               *m1.at( { x, y, z } ) = index++;
            }
         }
      }

      // sanity checks
      for ( size_t z = 0; z < size_m2[ 2 ]; ++z )
      {
         for ( size_t y = 0; y < size_m2[ 1 ]; ++y )
         {
            for ( size_t x = 0; x < size_m2[ 0 ]; ++x )
            {
               TESTER_ASSERT( *m2.at( { x, y, z } ) == *m1.at( NAMESPACE_NLL::vector3ui{ x, y, z } *stride_m2 + offset_start_m1 ) );
            }
         }
      }

      auto test_functor_dim = [&]( ui32 dim )
      {
         const NAMESPACE_NLL::vector3ui offset_start_m2 = { 1, 2, 3 };
         auto dstart = m2.beginDim( dim, offset_start_m2 );
         auto dend = m2.endDim( dim, offset_start_m2 );
         const auto nb_values = size_m2[ dim ] - offset_start_m2[ dim ];
         TESTER_ASSERT( nb_values == dend - dstart );

         TESTER_ASSERT( *dstart == *m1.at( offset_start_m2 * stride_m2 + offset_start_m1 ) );

         ui32 nb = 0;
         for ( auto it = dstart; it != dend; ++it, ++nb )
         {
            const auto diff = it - dstart;
            TESTER_ASSERT( diff == nb );
            TESTER_ASSERT( nb < nb_values ); // iterator did not stop as expected

            NAMESPACE_NLL::vector3ui expected_index = offset_start_m1 + offset_start_m2 * stride_m2;
            const ui32 offset = nb * stride_m2[ dim ];
            expected_index[ dim ] += offset;
            const auto expected_value = *m1.at( expected_index );
            const auto found = *it;
            TESTER_ASSERT( found == expected_value );
         }
      };

      test_functor_dim( 0 );
      test_functor_dim( 1 );
      test_functor_dim( 2 );
   }

   void testArray_processor()
   {
      testArray_processor_impl<NAMESPACE_NLL::Array_row_major<int, 3>>();
      testArray_processor_impl<NAMESPACE_NLL::Array_column_major<int, 3>>();
      testArray_processor_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 3>>();
   }

   template <class Array>
   void testArray_processor_impl()
   {
      const NAMESPACE_NLL::vector3ui size = { 40, 41, 42 };
      Array m1( size );
      int index = 0;
      for ( ui32 z = 0; z < size[ 2 ]; ++z )
      {
         for ( ui32 y = 0; y < size[ 1 ]; ++y )
         {
            for ( ui32 x = 0; x < size[ 0 ]; ++x )
            {
               m1( x, y, z ) = index++;
            }
         }
      }

      Array covered( size, 0 );

      NAMESPACE_NLL::details::ArrayProcessor_contiguous_base<Array> processor( m1, []( const Array& )
      {
         return NAMESPACE_NLL::vector3ui( 0, 1, 2 );
      } );
      bool has_more = true;
      while ( has_more )
      {
         int* value = 0;

         auto i = processor.getArrayIndex();
         has_more = processor.accessSingleElement( value );
         TESTER_ASSERT( m1( i ) == *value );

         covered( i ) = 1;
      }

      for ( ui32 z = 0; z < size[ 2 ]; ++z )
      {
         for ( ui32 y = 0; y < size[ 1 ]; ++y )
         {
            for ( ui32 x = 0; x < size[ 0 ]; ++x )
            {
               TESTER_ASSERT( covered( x, y, z ) == 1 ); // all voxels have been accessed
            }
         }
      }
   }

   void testArray_processor_stride()
   {
      testArray_processor_stride_impl<NAMESPACE_NLL::Array_row_major<int, 3>>();
      testArray_processor_stride_impl<NAMESPACE_NLL::Array_column_major<int, 3>>();
      testArray_processor_stride_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 3>>();
   }

   template <class Array>
   void testArray_processor_stride_impl()
   {
      const NAMESPACE_NLL::vector3ui size = { 40, 41, 42 };
      const NAMESPACE_NLL::vector3ui size_m2 = { 5, 6, 7 };
      const NAMESPACE_NLL::vector3ui stride_m2 = { 2, 1, 3 };
      const NAMESPACE_NLL::vector3ui offset_start_m1 = { 3, 1, 2 };

      auto functor = []( const Array& )
      {
         return NAMESPACE_NLL::vector3ui( 0, 1, 2 );
      };

      Array m1( size );
      Array m2( m1, offset_start_m1, size_m2, stride_m2 );

      // init
      int index = 0;
      for ( size_t z = 0; z < size[ 2 ]; ++z )
      {
         for ( size_t y = 0; y < size[ 1 ]; ++y )
         {
            for ( size_t x = 0; x < size[ 0 ]; ++x )
            {
               m1( x, y, z ) = index++;
            }
         }
      }

      // sanity checks
      for ( size_t z = 0; z < size_m2[ 2 ]; ++z )
      {
         for ( size_t y = 0; y < size_m2[ 1 ]; ++y )
         {
            for ( size_t x = 0; x < size_m2[ 0 ]; ++x )
            {
               TESTER_ASSERT( m2( x, y, z ) == m1( NAMESPACE_NLL::vector3ui{ x, y, z } *stride_m2 + offset_start_m1 ) );
            }
         }
      }

      Array covered( size_m2, 0 );

      // single access
      NAMESPACE_NLL::details::ArrayProcessor_contiguous_base<Array> processor( m2, functor );
      bool has_more = true;
      while ( has_more )
      {
         int* value = 0;

         auto i = processor.getArrayIndex();
         has_more = processor.accessSingleElement( value );
         TESTER_ASSERT( m2( i ) == *value );
      }

      // multiple accesses
      NAMESPACE_NLL::ArrayProcessor_contiguous_byMemoryLocality<Array> processor2( m2 );
      has_more = true;
      while ( has_more )
      {
         int* value = 0;

         auto i = processor2.getArrayIndex();
         has_more = processor2.accessMaxElements( value );

         const auto stride = processor2.stride() == 0 ? 1 : processor2.stride(); // TODO: not the best UT design, will fail for other types of Memory (eg., non linear)
         const auto maxElements = processor2.getMaxAccessElements();
         for ( ui32 n = 0; n < maxElements; ++n )
         {
            NAMESPACE_NLL::vector3ui index = i;
            index[ processor2.getVaryingIndex() ] += n;

            const auto value_value = *( value + n * stride );
            const auto expected_value = m2( index );
            TESTER_ASSERT( value_value == expected_value );
            covered( index ) = 1;
         }
      }

      for ( ui32 z = 0; z < size_m2[ 2 ]; ++z )
      {
         for ( ui32 y = 0; y < size_m2[ 1 ]; ++y )
         {
            for ( ui32 x = 0; x < size_m2[ 0 ]; ++x )
            {
               TESTER_ASSERT( covered( x, y, z ) == 1 ); // all voxels have been accessed
            }
         }
      }
   }

   void testIteratorByDim()
   {
      using array_type = NAMESPACE_NLL::Array_column_major < float, 3 > ;
      array_type a1( 4, 5, 6 );
      NAMESPACE_NLL::ArrayProcessor_contiguous_byDimension<array_type> iterator( a1 );
      TESTER_ASSERT( iterator.getVaryingIndexOrder() == core::vector3ui( 0, 1, 2 ) );
   }

   void testIteratorByLocality()
   {
      using array_type = NAMESPACE_NLL::Array_column_major < float, 3 >;
      array_type a1( 4, 5, 6 );
      NAMESPACE_NLL::ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator( a1 );
      TESTER_ASSERT( iterator.getVaryingIndexOrder() == core::vector3ui( 2, 1, 0 ) );
   }
};

TESTER_TEST_SUITE( TestArray );
TESTER_TEST( testVolumeConstruction_slices );
TESTER_TEST( testVolumeConstruction_slices_ref );
TESTER_TEST( testVolumeMove );
TESTER_TEST( testArray_construction );
TESTER_TEST( testMatrix_construction );
TESTER_TEST( testMatrix_construction2 );
TESTER_TEST( testArray_directional_index );
TESTER_TEST( testArray_processor );
TESTER_TEST( testArray_processor_stride );
TESTER_TEST( testIteratorByDim );
TESTER_TEST( testIteratorByLocality );
TESTER_TEST_SUITE_END();
