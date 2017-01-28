#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace nll;

using ui32 = NAMESPACE_NLL::ui32;

DECLARE_NAMESPACE_NLL
using vector3ui = StaticVector<ui32, 3>;
using vector2ui = StaticVector<ui32, 2>;
using vector1ui = StaticVector<ui32, 1>;

DECLARE_NAMESPACE_NLL_END

struct TestArray
{
   void testVolumeConstruction_slices()
   {
      NAMESPACE_NLL::Memory_multislice<float, 3> memory(NAMESPACE_NLL::vector3ui(2, 3, 4), 5);
      const auto& slices = memory._getSlices();
      TESTER_ASSERT(slices.size() == 4);
      TESTER_ASSERT(slices[1][2] == 5);

      auto p = memory.at({0, 0, 1});
      TESTER_ASSERT(*p == 5);
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
      const auto size = NAMESPACE_NLL::vector3ui(50, 51, 52);
      Memory memory(size, 0);

      int index = 0;
      for (size_t z = 0; z < size[2]; ++z)
      {
         for (size_t y = 0; y < size[1]; ++y)
         {
            for (size_t x = 0; x < size[0]; ++x)
            {
               *memory.at({x, y, z}) = index++;
            }
         }
      }

      const size_t in_slice_size = size[0] * size[1];

      TESTER_ASSERT(*memory.at({0, 0, 0}) == 0);
      TESTER_ASSERT(*memory.at({0, 0, 1}) == in_slice_size);
      TESTER_ASSERT(*memory.at({0, 0, 2}) == in_slice_size * 2);
      TESTER_ASSERT(*memory.at({1, 0, 2}) == in_slice_size * 2 + 1);
      TESTER_ASSERT(*memory.at({1, 2, 2}) == in_slice_size * 2 + 1 + 2 * size[0]);

      //
      // reference a volume
      //
      const NAMESPACE_NLL::vector3ui origin      = {2, 3, 4};
      const NAMESPACE_NLL::vector3ui sub_size    = {10, 11, 12};
      const NAMESPACE_NLL::vector3ui sub_strides = {2, 3, 2};
      Memory memory_ref(memory, origin, sub_size, sub_strides);
      {
         auto value    = *memory_ref.at({0, 0, 0});
         auto expected = *memory.at(origin);
         TESTER_ASSERT(value == expected);
      }

      for (int n = 0; n < 1000; ++n)
      {
         srand(n);
         const NAMESPACE_NLL::vector3ui displacement = {generateUniformDistribution<ui32>(0, sub_size[0] - 1),
                                                        generateUniformDistribution<ui32>(0, sub_size[1] - 1),
                                                        generateUniformDistribution<ui32>(0, sub_size[2] - 1)};
         auto value    = *memory_ref.at(displacement);
         auto expected = *memory.at(origin + displacement * sub_strides);
         TESTER_ASSERT(value == expected);
      }

      //
      // Reference a referenced volume with stride
      //
      const NAMESPACE_NLL::vector3ui origin2      = {2, 1, 0};
      const NAMESPACE_NLL::vector3ui sub_size2    = {2, 3, 2};
      const NAMESPACE_NLL::vector3ui sub_strides2 = {1, 3, 2};
      Memory memory_ref2(memory_ref, origin2, sub_size2, sub_strides2);

      for (int n = 0; n < 1000; ++n)
      {
         srand(n);
         const NAMESPACE_NLL::vector3ui displacement = {generateUniformDistribution<ui32>(0, sub_size2[0] - 1),
                                                        generateUniformDistribution<ui32>(0, sub_size2[1] - 1),
                                                        generateUniformDistribution<ui32>(0, sub_size2[2] - 1)};
         const auto value          = *memory_ref2.at(displacement);
         const auto index_original = (origin2 + (displacement)*sub_strides2) * sub_strides + origin;
         const auto expected       = *memory.at(index_original);
         TESTER_ASSERT(value == expected);
      }

      //
      // Test copy
      //
      Memory memory_cpy = memory_ref2;
      TESTER_ASSERT(memory_cpy.shape() == memory_ref2.shape());
      for (size_t n = 0; n < 500; ++n)
      {
         const NAMESPACE_NLL::vector3ui displacement = {generateUniformDistribution<ui32>(0, memory_ref2.shape()[0] - 1),
                                                        generateUniformDistribution<ui32>(0, memory_ref2.shape()[1] - 1),
                                                        generateUniformDistribution<ui32>(0, memory_ref2.shape()[2] - 1)};
         TESTER_ASSERT(*memory_cpy.at(displacement) == *memory_ref2.at(displacement));
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
      Memory test1({4, 5, 6});
      const auto ptr = test1.at({1, 2, 3});

      Memory test1_moved   = std::forward<Memory>(test1);
      const auto ptr_moved = test1_moved.at({1, 2, 3});

      TESTER_ASSERT(ptr_moved == ptr);

      Memory test2_moved;
      test2_moved           = std::forward<Memory>(test1_moved);
      const auto ptr_moved2 = test2_moved.at({1, 2, 3});
      TESTER_ASSERT(ptr_moved2 == ptr);
   }

   void testArray_construction()
   {
      using Array = NAMESPACE_NLL::Array<int, 3>;

      Array a1({1, 2, 3}, 42);
   }

   void testMatrix_construction()
   {
      NAMESPACE_NLL::Matrix<float> m({2, 3}, 0);
      TESTER_ASSERT(m.rows() == 2);
      TESTER_ASSERT(m.columns() == 3);

      m({1, 2}) = 42;
      TESTER_ASSERT(m(1, 2) == 42);
   }

   void testMatrix_construction2()
   {
      NAMESPACE_NLL::Matrix<float> m(2, 3); // construction with unpacked arugments
      TESTER_ASSERT(m.rows() == 2);
      TESTER_ASSERT(m.columns() == 3);
   }

   void testArray_directional_index()
   {
      testArray_directional_index_impl<NAMESPACE_NLL::Memory_contiguous<int, 3>>();
      testArray_directional_index_impl<NAMESPACE_NLL::Memory_multislice<int, 3>>();
   }

   template <class Memory>
   void testArray_directional_index_impl()
   {
      const NAMESPACE_NLL::vector3ui size            = {40, 41, 42};
      const NAMESPACE_NLL::vector3ui size_m2         = {5, 6, 7};
      const NAMESPACE_NLL::vector3ui stride_m2       = {2, 1, 3};
      const NAMESPACE_NLL::vector3ui offset_start_m1 = {3, 1, 2};

      Memory m1(size);
      Memory m2(m1, offset_start_m1, size_m2, stride_m2);

      // init
      int index = 0;
      for (size_t z = 0; z < size[2]; ++z)
      {
         for (size_t y = 0; y < size[1]; ++y)
         {
            for (size_t x = 0; x < size[0]; ++x)
            {
               *m1.at({x, y, z}) = index++;
            }
         }
      }

      // sanity checks
      for (size_t z = 0; z < size_m2[2]; ++z)
      {
         for (size_t y = 0; y < size_m2[1]; ++y)
         {
            for (size_t x = 0; x < size_m2[0]; ++x)
            {
               TESTER_ASSERT(*m2.at({x, y, z}) == *m1.at(NAMESPACE_NLL::vector3ui{x, y, z} * stride_m2 + offset_start_m1));
            }
         }
      }

      auto test_functor_dim = [&](ui32 dim) {
         const NAMESPACE_NLL::vector3ui offset_start_m2 = {1, 2, 3};
         auto dstart                                    = m2.beginDim(dim, offset_start_m2);
         auto dend                                      = m2.endDim(dim, offset_start_m2);
         const auto nb_values                           = size_m2[dim] - offset_start_m2[dim];
         TESTER_ASSERT(nb_values == dend - dstart);

         TESTER_ASSERT(*dstart == *m1.at(offset_start_m2 * stride_m2 + offset_start_m1));

         ui32 nb = 0;
         for (auto it = dstart; it != dend; ++it, ++nb)
         {
            const auto diff = it - dstart;
            TESTER_ASSERT(diff == nb);
            TESTER_ASSERT(nb < nb_values); // iterator did not stop as expected

            NAMESPACE_NLL::vector3ui expected_index = offset_start_m1 + offset_start_m2 * stride_m2;
            const ui32 offset                       = nb * stride_m2[dim];
            expected_index[dim] += offset;
            const auto expected_value = *m1.at(expected_index);
            const auto found          = *it;
            TESTER_ASSERT(found == expected_value);
         }
      };

      test_functor_dim(0);
      test_functor_dim(1);
      test_functor_dim(2);
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
      const NAMESPACE_NLL::vector3ui size = {40, 41, 42};
      Array m1(size);
      int index = 0;
      for (ui32 z = 0; z < size[2]; ++z)
      {
         for (ui32 y = 0; y < size[1]; ++y)
         {
            for (ui32 x = 0; x < size[0]; ++x)
            {
               m1(x, y, z) = index++;
            }
         }
      }

      Array covered(size, 0);

      NAMESPACE_NLL::details::ArrayProcessor_contiguous_base<Array> processor(m1, [](const Array&) { return NAMESPACE_NLL::vector3ui(0, 1, 2); });
      bool has_more = true;
      while (has_more)
      {
         int* value = 0;

         auto i   = processor.getArrayIndex();
         has_more = processor.accessSingleElement(value);
         TESTER_ASSERT(m1(i) == *value);

         covered(i) = 1;
      }

      for (ui32 z = 0; z < size[2]; ++z)
      {
         for (ui32 y = 0; y < size[1]; ++y)
         {
            for (ui32 x = 0; x < size[0]; ++x)
            {
               TESTER_ASSERT(covered(x, y, z) == 1); // all voxels have been accessed
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
      const NAMESPACE_NLL::vector3ui size            = {40, 41, 42};
      const NAMESPACE_NLL::vector3ui size_m2         = {5, 6, 7};
      const NAMESPACE_NLL::vector3ui stride_m2       = {2, 1, 3};
      const NAMESPACE_NLL::vector3ui offset_start_m1 = {3, 1, 2};

      auto functor = [](const Array&) { return NAMESPACE_NLL::vector3ui(0, 1, 2); };

      Array m1(size);
      Array m2(m1, offset_start_m1, size_m2, stride_m2);

      // init
      int index = 0;
      for (size_t z = 0; z < size[2]; ++z)
      {
         for (size_t y = 0; y < size[1]; ++y)
         {
            for (size_t x = 0; x < size[0]; ++x)
            {
               m1(x, y, z) = index++;
            }
         }
      }

      // sanity checks
      for (size_t z = 0; z < size_m2[2]; ++z)
      {
         for (size_t y = 0; y < size_m2[1]; ++y)
         {
            for (size_t x = 0; x < size_m2[0]; ++x)
            {
               TESTER_ASSERT(m2(x, y, z) == m1(NAMESPACE_NLL::vector3ui{x, y, z} * stride_m2 + offset_start_m1));
            }
         }
      }

      Array covered(size_m2, 0);

      // single access
      NAMESPACE_NLL::details::ArrayProcessor_contiguous_base<Array> processor(m2, functor);
      bool has_more = true;
      while (has_more)
      {
         int* value = 0;

         auto i   = processor.getArrayIndex();
         has_more = processor.accessSingleElement(value);
         TESTER_ASSERT(m2(i) == *value);
      }

      // multiple accesses
      NAMESPACE_NLL::ArrayProcessor_contiguous_byMemoryLocality<Array> processor2(m2);
      has_more = true;
      while (has_more)
      {
         int* value = 0;

         auto i   = processor2.getArrayIndex();
         has_more = processor2.accessMaxElements(value);

         const auto stride =
             processor2.stride() == 0 ? 1 : processor2.stride(); // TODO: not the best UT design, will fail for other types of Memory (eg., non linear)
         const auto maxElements = processor2.getMaxAccessElements();
         for (ui32 n = 0; n < maxElements; ++n)
         {
            NAMESPACE_NLL::vector3ui index = i;
            index[processor2.getVaryingIndex()] += n;

            const auto value_value    = *(value + n * stride);
            const auto expected_value = m2(index);
            TESTER_ASSERT(value_value == expected_value);
            covered(index) = 1;
         }
      }

      for (ui32 z = 0; z < size_m2[2]; ++z)
      {
         for (ui32 y = 0; y < size_m2[1]; ++y)
         {
            for (ui32 x = 0; x < size_m2[0]; ++x)
            {
               TESTER_ASSERT(covered(x, y, z) == 1); // all voxels have been accessed
            }
         }
      }
   }

   void testIteratorByDim()
   {
      using array_type = NAMESPACE_NLL::Array_column_major<float, 3>;
      array_type a1(4, 5, 6);
      NAMESPACE_NLL::ArrayProcessor_contiguous_byDimension<array_type> iterator(a1);
      TESTER_ASSERT(iterator.getVaryingIndexOrder() == NAMESPACE_NLL::vector3ui(0, 1, 2));
   }

   void testIteratorByLocality()
   {
      using array_type = NAMESPACE_NLL::Array_column_major<float, 3>;
      array_type a1(4, 5, 6);
      NAMESPACE_NLL::ArrayProcessor_contiguous_byMemoryLocality<array_type> iterator(a1);
      TESTER_ASSERT(iterator.getVaryingIndexOrder() == NAMESPACE_NLL::vector3ui(2, 1, 0));
   }

   void testFill()
   {
      // not a real test, just for very simplistic performace test against manual op
      using array_type = NAMESPACE_NLL::Array_row_major<int, 2>;
      array_type a1(60, 40);

      auto functor = [](const NAMESPACE_NLL::StaticVector<ui32, 2>& index) { return (int)(index[0] * index[1]); };

      NAMESPACE_NLL::fill(a1, functor);

      TESTER_ASSERT(a1(2, 3) == 2 * 3);
      TESTER_ASSERT(a1(1, 2) == 1 * 2);
   }

   void testArray_subArray()
   {
      testArray_subArray_impl<NAMESPACE_NLL::Array_row_major<int, 3>>();
      testArray_subArray_impl<NAMESPACE_NLL::Array_column_major<int, 3>>();
      testArray_subArray_impl<NAMESPACE_NLL::Array_row_major_multislice<int, 3>>();
   }

   template <class array_type>
   void testArray_subArray_impl()
   {
      array_type a1(20, 30, 40);

      int index = 0;
      NAMESPACE_NLL::fill(a1, [&](const NAMESPACE_NLL::vector3ui&) { return index++; });

      const NAMESPACE_NLL::vector3ui min_index(4, 5, 6);
      const NAMESPACE_NLL::vector3ui max_index(8, 10, 16);
      auto ref_a1 = a1(min_index, max_index);

      const auto expected_size = max_index - min_index + NAMESPACE_NLL::vector3ui(1);

      const NAMESPACE_NLL::vector3ui min_index2(8, 10, 15);
      auto ref_a2 = a1(min_index2, min_index2 + expected_size - 1);

      TESTER_ASSERT(expected_size == ref_a1.shape());
      TESTER_ASSERT(ref_a1(0, 0, 0) == a1(min_index));
      TESTER_ASSERT(ref_a1(max_index - min_index) == a1(max_index));

      TESTER_ASSERT(ref_a2.shape() == ref_a1.shape());

      ref_a2 = ref_a1;
   }

   void testInitializerList()
   {
      testInitializerList_impl<NAMESPACE_NLL::Array_column_major<int, 2>>();
      testInitializerList_impl<NAMESPACE_NLL::Array_row_major<int, 2>>();
   }

   template <class Array>
   void testInitializerList_impl()
   {
      Array m(3, 2);
      m = {1, 2, 3, 4, 5, 6};
      TESTER_ASSERT(m(0, 0) == 1);
      TESTER_ASSERT(m(1, 0) == 2);
      TESTER_ASSERT(m(2, 0) == 3);
      TESTER_ASSERT(m(0, 1) == 4);
      TESTER_ASSERT(m(1, 1) == 5);
      TESTER_ASSERT(m(2, 1) == 6);
   }

   void testIndeMapperRebind()
   {
      using Array          = NAMESPACE_NLL::Array_column_major<int, 3>;
      using NewIndexMapper = Array::Memory::index_mapper::rebind<2>::other;
      static_assert(std::is_same<NewIndexMapper, NAMESPACE_NLL::IndexMapper_contiguous_column_major<2>>::value, "must be the same!");

      //using MemoryRebind = Array::Memory::rebind<int, 2>;
   }

   
   void testMemorySlice()
   {
      testMemorySlice_impl<NAMESPACE_NLL::Array_row_major<short, 2>>();
      testMemorySlice_impl<NAMESPACE_NLL::Array_column_major<float, 2>>();
   }

   template <class Array>
   void testMemorySlice_impl()
   {
      Array m(2, 3);
      m = {1, 2, 3, 4, 5, 6};

      {
         auto sliced = m.getMemory().template slice<0>(NAMESPACE_NLL::vector2ui{ 1, 0 });
         TESTER_ASSERT(sliced.shape() == NAMESPACE_NLL::vector1ui{3});
         TESTER_ASSERT(*sliced.at({0}) == 2);
         TESTER_ASSERT(*sliced.at({1}) == 4);
         TESTER_ASSERT(*sliced.at({2}) == 6);
      }

      {
         auto sliced = m.getMemory().template slice<1>(NAMESPACE_NLL::vector2ui{ 0, 2 });
         TESTER_ASSERT(sliced.shape() == NAMESPACE_NLL::vector1ui{2});
         TESTER_ASSERT(*sliced.at({0}) == 5);
         TESTER_ASSERT(*sliced.at({1}) == 6);
      }
   }

   void testMemorySliceNonContiguous_zslice()
   {
      testMemorySliceNonContiguous_zslice_impl<NAMESPACE_NLL::Array_row_major<short, 3>>();
      testMemorySliceNonContiguous_zslice_impl<NAMESPACE_NLL::Array_column_major<short, 3>>();
      testMemorySliceNonContiguous_zslice_impl<NAMESPACE_NLL::Array_row_major_multislice<short, 3>>();
   }

   template <class Array>
   void testMemorySliceNonContiguous_zslice_impl()
   {
      const size_t dim = 2;

      for (size_t nn = 0; nn < 100; ++nn)
      {
         srand((unsigned)nn);
         NAMESPACE_NLL::vector3ui size(generateUniformDistribution(1, 20), generateUniformDistribution(1, 20), generateUniformDistribution(1, 20));

         Array a1(size);
         short index = 0;
         NAMESPACE_NLL::fill(a1, [&](const NAMESPACE_NLL::vector3ui&) { return index++; });

         const auto origin = NAMESPACE_NLL::vector3ui{generateUniformDistribution<size_t>(0, size[0] - 1), generateUniformDistribution<size_t>(0, size[1] - 1),
                                                      generateUniformDistribution<size_t>(0, size[2] - 1)};

         auto& memory = a1.getMemory();
         auto sliced  = memory.template slice<2>(origin);

         TESTER_ASSERT(a1({0, 0, origin[dim]}) == *sliced.at({0, 0}));

         for (size_t y = 0; y < size[1]; ++y)
         {
            for (size_t x = 0; x < size[0]; ++x)
            {
               const auto value_expected = a1(NAMESPACE_NLL::vector3ui(x, y, origin[dim]));
               const auto value_found    = *sliced.at({x, y});
               TESTER_ASSERT(value_expected == value_found);
            }
         }
      }
   }

   void testMemorySliceNonContiguous_not_zslice()
   {
      testMemorySliceNonContiguous_not_zslice_impl<NAMESPACE_NLL::Array_row_major<short, 3>>();
      testMemorySliceNonContiguous_not_zslice_impl<NAMESPACE_NLL::Array_column_major<short, 3>>();
      testMemorySliceNonContiguous_not_zslice_impl<NAMESPACE_NLL::Array_row_major_multislice<short, 3>>();
   }

   template <class Array>
   void testMemorySliceNonContiguous_not_zslice_impl()
   {
      const size_t dim = 0;

      for (size_t nn = 0; nn < 100; ++nn)
      {
         srand((unsigned)nn + 1);
         NAMESPACE_NLL::vector3ui size(generateUniformDistribution(1, 20), generateUniformDistribution(1, 20), generateUniformDistribution(1, 20));

         Array a1(size);
         short index = 0;
         NAMESPACE_NLL::fill(a1, [&](const NAMESPACE_NLL::vector3ui&) { return index++; });

         const auto origin = NAMESPACE_NLL::vector3ui{generateUniformDistribution<size_t>(0, size[0] - 1), generateUniformDistribution<size_t>(0, size[1] - 1),
                                                      generateUniformDistribution<size_t>(0, size[2] - 1)};

         auto& memory = a1.getMemory();
         auto sliced  = memory.template slice<(0)>(origin);

         TESTER_ASSERT(a1({origin[dim], 0, 0}) == *sliced.at({0, 0}));

         for (size_t z = 0; z < size[2]; ++z)
         {
            for (size_t y = 0; y < size[1]; ++y)
            {
               const auto value_expected = a1(NAMESPACE_NLL::vector3ui(origin[dim], y, z));
               const auto value_found    = *sliced.at({y, z});
               TESTER_ASSERT(value_expected == value_found);
            }
         }
      }
   }
   
   
   void testArraySlice()
   {
      testArraySlice_impl<NAMESPACE_NLL::Array_row_major<short, 3>>();
      testArraySlice_impl<NAMESPACE_NLL::Array_column_major<short, 3>>();
      testArraySlice_impl<NAMESPACE_NLL::Array_row_major_multislice<short, 3>>();
   }

   template <class Array>
   void testArraySlice_impl()
   {
      srand((unsigned)1);
      NAMESPACE_NLL::vector3ui size(generateUniformDistribution(5, 20), generateUniformDistribution(5, 20), generateUniformDistribution(5, 20));

      Array a1(size);
      short index = 0;
      NAMESPACE_NLL::fill(a1, [&](const NAMESPACE_NLL::vector3ui&) { return index++; });

      // just to clarify if there is a problem with the template here...
      using SlicedMemory = typename Array::template SlicingMemory<2>;
      using Sliced = typename Array::template SlicingArray<2>;
      
      const int slice_a = 1;
      auto slice = a1.template slice<2>(NAMESPACE_NLL::vector3ui{ 2, 3, slice_a });
      
      TESTER_ASSERT(slice.shape() == NAMESPACE_NLL::vector2ui(a1.shape()[0], a1.shape()[1]));
      TESTER_ASSERT(slice(0, 0) == a1(0, 0, slice_a));
      TESTER_ASSERT(slice(1, 0) == a1(1, 0, slice_a));
      TESTER_ASSERT(slice(0, 1) == a1(0, 1, slice_a));

      const int slice_b = 2;
      auto slice2 = slice.template slice<1>(NAMESPACE_NLL::vector2ui{ 0, slice_b });
      TESTER_ASSERT(slice2(0) == a1(0, slice_b, slice_a));
      TESTER_ASSERT(slice2(1) == a1(1, slice_b, slice_a));
      TESTER_ASSERT(slice2(2) == a1(2, slice_b, slice_a));
   }

   void testArray_processor_const()
   {
      using Array = NAMESPACE_NLL::Array_row_major<int, 2>;
      Array m(3, 2);
      m = {1, 2, 3, 4, 5, 6};

      NAMESPACE_NLL::ConstArrayProcessor_contiguous_byMemoryLocality<Array> processor(m);
      TESTER_ASSERT(processor.getVaryingIndex() == 0);

      bool more_elements = true;
      int const* ptr     = nullptr;

      auto current_index = processor.getArrayIndex();
      more_elements      = processor.accessMaxElements(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 1);
      TESTER_ASSERT(ptr[1] == 2);
      TESTER_ASSERT(ptr[2] == 3);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessMaxElements(ptr);
      TESTER_ASSERT(!more_elements);
      TESTER_ASSERT(ptr[0] == 4);
      TESTER_ASSERT(ptr[1] == 5);
      TESTER_ASSERT(ptr[2] == 6);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 1));
   }

   void testArray_processor_const_column()
   {
      using Array = NAMESPACE_NLL::Array_column_major<int, 2>;
      Array m(2, 3);
      m = {1, 2, 3, 4, 5, 6};

      NAMESPACE_NLL::ConstArrayProcessor_contiguous_byMemoryLocality<Array> processor(m);
      TESTER_ASSERT(processor.getVaryingIndex() == 1);

      bool more_elements = true;
      int const* ptr     = nullptr;

      auto current_index = processor.getArrayIndex();
      more_elements      = processor.accessMaxElements(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 1);
      TESTER_ASSERT(ptr[1] == 3);
      TESTER_ASSERT(ptr[2] == 5);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessMaxElements(ptr);
      TESTER_ASSERT(!more_elements);
      TESTER_ASSERT(ptr[0] == 2);
      TESTER_ASSERT(ptr[1] == 4);
      TESTER_ASSERT(ptr[2] == 6);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 0));

      //ptr[ 0 ] = 42;
   }

   void testMemory_processor_const_column()
   {
      using Array = NAMESPACE_NLL::Array_column_major<int, 2>;
      Array m(2, 3);
      m = {1, 2, 3, 4, 5, 6};

      NAMESPACE_NLL::ConstMemoryProcessor_contiguous_byMemoryLocality<Array::Memory> processor(m.getMemory());
      TESTER_ASSERT(processor.getVaryingIndex() == 1);

      bool more_elements = true;
      int const* ptr     = nullptr;

      auto current_index = processor.getArrayIndex();
      more_elements      = processor.accessMaxElements(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 1);
      TESTER_ASSERT(ptr[1] == 3);
      TESTER_ASSERT(ptr[2] == 5);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessMaxElements(ptr);
      TESTER_ASSERT(!more_elements);
      TESTER_ASSERT(ptr[0] == 2);
      TESTER_ASSERT(ptr[1] == 4);
      TESTER_ASSERT(ptr[2] == 6);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 0));

      //ptr[ 0 ] = 42;
   }

   void testMemory_processor_column()
   {
      using Array = NAMESPACE_NLL::Array_column_major<int, 2>;
      Array m(2, 3);
      m = {1, 2, 3, 4, 5, 6};

      NAMESPACE_NLL::MemoryProcessor_contiguous_byMemoryLocality<Array::Memory> processor(m.getMemory());
      TESTER_ASSERT(processor.getVaryingIndex() == 1);

      bool more_elements = true;
      int* ptr           = nullptr;

      auto current_index = processor.getArrayIndex();
      more_elements      = processor.accessMaxElements(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 1);
      TESTER_ASSERT(ptr[1] == 3);
      TESTER_ASSERT(ptr[2] == 5);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessMaxElements(ptr);
      TESTER_ASSERT(!more_elements);
      TESTER_ASSERT(ptr[0] == 2);
      TESTER_ASSERT(ptr[1] == 4);
      TESTER_ASSERT(ptr[2] == 6);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 0));

      //ptr[ 0 ] = 42;
   }

   void testArray_processor_const_column2()
   {
      using Array = NAMESPACE_NLL::Array_column_major<int, 2>;
      Array m(2, 3);
      m = {1, 2, 3, 4, 5, 6};

      NAMESPACE_NLL::ConstArrayProcessor_contiguous_byDimension<Array> processor(m);
      TESTER_ASSERT(processor.getVaryingIndex() == 0);

      bool more_elements = true;
      int const* ptr     = nullptr;

      auto current_index = processor.getArrayIndex();
      more_elements      = processor.accessSingleElement(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 1);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessSingleElement(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 2);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 0));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessSingleElement(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 3);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 1));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessSingleElement(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 4);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 1));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessSingleElement(ptr);
      TESTER_ASSERT(more_elements);
      TESTER_ASSERT(ptr[0] == 5);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(0, 2));

      current_index = processor.getArrayIndex();
      more_elements = processor.accessSingleElement(ptr);
      TESTER_ASSERT(!more_elements);
      TESTER_ASSERT(ptr[0] == 6);
      TESTER_ASSERT(current_index == NAMESPACE_NLL::vector2ui(1, 2));
   }

   void testStaticCast()
   {
      using Array = NAMESPACE_NLL::Array_column_major<float, 2>;
      Array m(2, 3);
      m = { 1.1f, 2.0f, 3.6f, 4.9f, 5.0f, 6.0f };

      const auto casted = m.staticCastTo<int>();
      TESTER_ASSERT(casted(0, 0) == 1);
      TESTER_ASSERT(casted(1, 0) == 2);
      TESTER_ASSERT(casted(0, 1) == 3);
      TESTER_ASSERT(casted(1, 1) == 4);
      TESTER_ASSERT(casted(0, 2) == 5);
      TESTER_ASSERT(casted(1, 2) == 6);
   }

   void testStaticCast_slice()
   {
      using Array = NAMESPACE_NLL::Array_row_major_multislice<float, 2>;
      Array m(2, 3);
      m = { 1.1f, 2.0f, 3.6f, 4.9f, 5.0f, 6.0f };

      const auto casted = m.staticCastTo<int>();
      TESTER_ASSERT(casted(0, 0) == 1);
      TESTER_ASSERT(casted(1, 0) == 2);
      TESTER_ASSERT(casted(0, 1) == 3);
      TESTER_ASSERT(casted(1, 1) == 4);
      TESTER_ASSERT(casted(0, 2) == 5);
      TESTER_ASSERT(casted(1, 2) == 6);
   }
};

TESTER_TEST_SUITE(TestArray);
TESTER_TEST(testMemory_processor_const_column);
TESTER_TEST(testMemory_processor_column);
TESTER_TEST(testVolumeConstruction_slices);
TESTER_TEST(testVolumeConstruction_slices_ref);
TESTER_TEST(testVolumeMove);
TESTER_TEST(testArray_construction);
TESTER_TEST(testMatrix_construction);
TESTER_TEST(testMatrix_construction2);
TESTER_TEST(testArray_directional_index);
TESTER_TEST(testArray_processor);
TESTER_TEST(testArray_processor_const);
TESTER_TEST(testArray_processor_stride);
TESTER_TEST(testIteratorByDim);
TESTER_TEST(testIteratorByLocality);
TESTER_TEST(testFill);
TESTER_TEST(testArray_subArray);
TESTER_TEST(testInitializerList);
TESTER_TEST(testIndeMapperRebind);
TESTER_TEST(testMemorySlice);
TESTER_TEST(testArraySlice);
TESTER_TEST(testMemorySliceNonContiguous_not_zslice);
TESTER_TEST(testMemorySliceNonContiguous_zslice);
TESTER_TEST(testArray_processor_const_column);
TESTER_TEST(testArray_processor_const_column2);
TESTER_TEST(testStaticCast);
TESTER_TEST(testStaticCast_slice);
TESTER_TEST_SUITE_END();
