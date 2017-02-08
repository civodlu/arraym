#pragma warning(disable : 4244)

#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;

using vector5ui = StaticVector<ui32, 5>;

struct TestIndexMapper
{
   void test_column_major_mapper()
   {
      details::Mapper_stride_column_major<3> im;
      const auto stride = im(vector3ui(2, 3, 4));
      TESTER_ASSERT(stride[2] == 1);
      TESTER_ASSERT(stride[1] == 4);
      TESTER_ASSERT(stride[0] == 12);
   }

   void test_row_major_mapper()
   {
      details::Mapper_stride_row_major<3> im;
      const auto stride = im(vector3ui(2, 3, 4));
      TESTER_ASSERT(stride[0] == 1);
      TESTER_ASSERT(stride[1] == 2);
      TESTER_ASSERT(stride[2] == 6);
   }

   void test_row_major_mapper_extend()
   {
      details::Mapper_stride_row_major<3> im3;
      const auto stride = im3(vector3ui(2, 3, 4));


      details::Mapper_stride_row_major<5> im5;
      vector5ui stride5(stride[0], stride[1], stride[2], 0, 0);
      im5.extend_stride(stride5, vector5ui(2, 3, 4, 5, 6), 3, 2);
      
      TESTER_ASSERT(stride5[0] == stride[0]);
      TESTER_ASSERT(stride5[1] == stride[1]);
      TESTER_ASSERT(stride5[2] == stride[2]);
      TESTER_ASSERT(stride5[3] == stride[2] * 4);
      TESTER_ASSERT(stride5[4] == stride[2] * 5 * 4);
   }

   void test_row_column_mapper_extend()
   {
      details::Mapper_stride_column_major<3> im3;
      const auto stride = im3(vector3ui(2, 3, 4));


      details::Mapper_stride_column_major<5> im5;
      vector5ui stride5(stride[0], stride[1], stride[2], 0, 0);
      im5.extend_stride(stride5, vector5ui(2, 3, 4, 5, 6), 3, 0);

      TESTER_ASSERT(stride5[0] == stride[0]);
      TESTER_ASSERT(stride5[1] == stride[1]);
      TESTER_ASSERT(stride5[2] == stride[2]);
      TESTER_ASSERT(stride5[3] == stride[0] * 6 * 5);
      TESTER_ASSERT(stride5[4] == stride[0] * 6);
   }
};

TESTER_TEST_SUITE(TestIndexMapper);
TESTER_TEST(test_column_major_mapper);
TESTER_TEST(test_row_major_mapper);
TESTER_TEST(test_row_major_mapper_extend);
TESTER_TEST(test_row_column_mapper_extend);
TESTER_TEST_SUITE_END();