
#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using vector3ui = StaticVector<ui32, 3>;
using vector2ui = StaticVector<ui32, 2>;
using vector1ui = StaticVector<ui32, 1>;

struct TestArrayChunking
{
   void test_rowMajor()
   {
      using Array = Array_row_major<float, 2>;

      Array a(2, 3);
      a = { 1, 2, 3, 4, 5, 6 };

      ArrayChunking_contiguous_base<Array> chunking(a, [](const Array&){return vector2ui(0, 1); });

      bool done;
      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 0));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 0));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 1));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 1));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 2));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);
      
      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 2));
      done = chunking._accessElements(1);
      TESTER_ASSERT(!done);
   }

   void test_colMajor()
   {
      using Array = Array_row_major<float, 2>;

      Array a(2, 3);
      a = { 1, 2, 3, 4, 5, 6 };

      ArrayChunking_contiguous_base<Array> chunking(a, [](const Array&){return vector2ui(1, 0); });

      bool done;
      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 0));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 1));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 2));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 0));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 1));
      done = chunking._accessElements(1);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 2));
      done = chunking._accessElements(1);
      TESTER_ASSERT(!done);
   }

   void test_rowMajor_max()
   {
      using Array = Array_row_major<float, 2>;

      Array a(2, 3);
      a = { 1, 2, 3, 4, 5, 6 };

      ArrayChunking_contiguous_base<Array> chunking(a, [](const Array&){return vector2ui(0, 1); });

      bool done;
      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 0));
      done = chunking._accessElements(2);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 1));
      done = chunking._accessElements(2);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 2));
      done = chunking._accessElements(2);
      TESTER_ASSERT(!done);
   }

   void test_colMajor_max()
   {
      using Array = Array_column_major<float, 2>;

      Array a(2, 3);
      a = { 1, 2, 3, 4, 5, 6 };

      ArrayChunking_contiguous_base<Array> chunking(a, [](const Array&){return vector2ui(1, 0); });

      bool done;
      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(0, 0));
      done = chunking._accessElements(3);
      TESTER_ASSERT(done);

      TESTER_ASSERT(chunking.getArrayIndex() == vector2ui(1, 0));
      done = chunking._accessElements(3);
      TESTER_ASSERT(!done);
   }
};

TESTER_TEST_SUITE(TestArrayChunking);
TESTER_TEST(test_rowMajor);
TESTER_TEST(test_rowMajor_max);
TESTER_TEST(test_colMajor);
TESTER_TEST(test_colMajor_max);
TESTER_TEST_SUITE_END();