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
};

TESTER_TEST_SUITE( TestEnumerate );
TESTER_TEST( test_enumerate_vectors );
TESTER_TEST( test_enumerate_cont_vectors );
TESTER_TEST_SUITE_END();