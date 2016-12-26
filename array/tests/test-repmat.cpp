#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector1ui = StaticVector<ui32, 1>;
using vector2ui = StaticVector<ui32, 2>;
using vector3ui = StaticVector<ui32, 3>;

template <class T, size_t N, class Config, size_t N2>
Array<T, N2, typename Config::template rebind_dim<N2>::other> repmat(const Array<T, N, Config>& array, const StaticVector<ui32, N2>& times)
{
   static_assert(N2 >= N, "N2 must have at least the same dimension as the array");
   using other_array_type = Array<T, N2, typename Config::template rebind_dim<N2>::other>;
   using array_type = Array<T, N, Config>;

   // first convert to the correct dimention
   typename other_array_type::index_type shape_n2;
   typename other_array_type::index_type other_shape;
   for (size_t n = 0; n < N; ++n)
   {
      shape_n2[n] = array.shape()[n];
      other_shape[n] = array.shape()[n] * times[n];
   }
   for (size_t n = N; n < N2; ++n)
   {
      shape_n2[n] = 1;
      other_shape[n] = times[n];
   }
   const auto array_n2 = as_array(array, shape_n2);

   // then iterate array in a memory friendly manner
   other_array_type other(other_shape);
   ArrayChunking_contiguous_base<other_array_type> chunking(times, getFastestVaryingIndexes(other));
   
   bool more_elements = true;
   while (more_elements)
   {
      const auto index = chunking.getArrayIndex() * shape_n2;
      more_elements = chunking._accessElements(1);
      other(index, index + shape_n2 - 1) = array_n2;
   }
   
   return other;
}

struct TestArrayRepmat
{
   void testRepmat_simple()
   {
      Array<int, 1> vec(3);
      vec = { 1, 2, 3};

      auto vec32 = repmat(vec, vector2ui(1, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(3, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);
   }

   void testRepmat_simple2()
   {
      Array<int, 1> vec(3);
      vec = { 1, 2, 3 };

      auto vec32 = repmat(vec, vector2ui(2, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(6, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);

      TESTER_ASSERT(vec32(0 + 3, 0) == 1);
      TESTER_ASSERT(vec32(1 + 3, 0) == 2);
      TESTER_ASSERT(vec32(2 + 3, 0) == 3);
      TESTER_ASSERT(vec32(0 + 3, 1) == 1);
      TESTER_ASSERT(vec32(1 + 3, 1) == 2);
      TESTER_ASSERT(vec32(2 + 3, 1) == 3);
   }

   void testRepmat_simple3()
   {
      Array<int, 1> vec(3);
      vec = { 1, 2, 3 };

      auto vec32 = repmat(vec, vector3ui(1, 2, 2));
      TESTER_ASSERT(vec32.shape() == vector3ui(3, 2, 2));
      TESTER_ASSERT(vec32(0, 0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0, 0) == 2);
      TESTER_ASSERT(vec32(2, 0, 0) == 3);
      TESTER_ASSERT(vec32(0, 1, 0) == 1);
      TESTER_ASSERT(vec32(1, 1, 0) == 2);
      TESTER_ASSERT(vec32(2, 1, 0) == 3);

      TESTER_ASSERT(vec32(0, 0, 1) == 1);
      TESTER_ASSERT(vec32(1, 0, 1) == 2);
      TESTER_ASSERT(vec32(2, 0, 1) == 3);
      TESTER_ASSERT(vec32(0, 1, 1) == 1);
      TESTER_ASSERT(vec32(1, 1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1, 1) == 3);
   }

   void testRepmat_strided()
   {
      int values[] =
      {
         1, 0, 2, 0, 3, 0
      };

      using array_type = Array<int, 1>;
      array_type vec(array_type::Memory(vector1ui(3), values, vector1ui(2)));
      TESTER_ASSERT(vec(0) == 1);
      TESTER_ASSERT( vec( 1 ) == 2 );
      TESTER_ASSERT(vec(2) == 3);

      auto vec32 = repmat(vec, vector2ui(1, 2));
      TESTER_ASSERT(vec32.shape() == vector2ui(3, 2));
      TESTER_ASSERT(vec32(0, 0) == 1);
      TESTER_ASSERT(vec32(1, 0) == 2);
      TESTER_ASSERT(vec32(2, 0) == 3);
      TESTER_ASSERT(vec32(0, 1) == 1);
      TESTER_ASSERT(vec32(1, 1) == 2);
      TESTER_ASSERT(vec32(2, 1) == 3);
   }
};

TESTER_TEST_SUITE(TestArrayRepmat);
TESTER_TEST(testRepmat_simple);
TESTER_TEST(testRepmat_simple2);
TESTER_TEST(testRepmat_simple3);
TESTER_TEST(testRepmat_strided);
TESTER_TEST_SUITE_END();