#include <array/forward.h>
#include <tester/register.h>

using namespace NAMESPACE_NLL;


struct TestEnumerate
{
   void test_enumerate_vectors()
   {
      std::vector<ui32> vec = { 2, 4, 6, 8 };
      
      ui32 index = 0;
      for ( auto& it : enumerate( vec ) )
      {
         std::cout << it.index << " " << *it.iterator << std::endl;
         TESTER_ASSERT( index == it.index );
         TESTER_ASSERT( vec[index] == *it );

         *it = index;
         TESTER_ASSERT( vec[ index ] == index );
         ++index;
      }
   }
   
   void test_enumerate_cont_vectors()
   {
      std::vector<int> vec = { 2, 4, 6, 8 };

      const std::vector<int>& vec_const = vec;

      ui32 index = 0;
      for ( auto& it : enumerate( vec_const ) )
      {
         std::cout << it.index << " " << *it.iterator << std::endl;
         TESTER_ASSERT( index == it.index );
         TESTER_ASSERT( vec_const[ index ] == *it );
         ++index;
      }
   }

   void test_enumerate_const_rows()
   {
      using array_type = Array<float, 2>;

      array_type array(2, 3);
      array = { 1, 2,
         3, 4,
         5, 6 };

      const array_type& array_const = array;

      ui32 index = 0;
      for (auto it : enumerate(rows(array_const)))
      {
         TESTER_ASSERT(it.index == index);
         TESTER_ASSERT((*it)({ 0, 0 }) == array({ 0, index }));
         TESTER_ASSERT((*it)({ 1, 0 }) == array({ 1, index }));
         ++index;
      }
      TESTER_ASSERT(index == 3);
   }
};

TESTER_TEST_SUITE( TestEnumerate );
TESTER_TEST( test_enumerate_vectors );
TESTER_TEST( test_enumerate_cont_vectors );
TESTER_TEST(test_enumerate_const_rows);
TESTER_TEST_SUITE_END();