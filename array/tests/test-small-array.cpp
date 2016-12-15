#include <array/forward.h>
#include <tester/register.h>
#include "test-utils.h"

using namespace NAMESPACE_NLL;

using vector2ui = StaticVector<ui32, 2>;

struct TestSmallArray
{
   void test_small_vector()
   {
      using V = Vector<int, AllocatorSingleStaticMemory<int, 4>>;

      V t1( 4 );
      t1 = { 1, 2, 3, 4 };

      V t2 = t1;
      TESTER_ASSERT( &t1( 0 ) != &t2( 0 ) );
      TESTER_ASSERT( t2( 0 ) == 1 );
      TESTER_ASSERT( t2( 1 ) == 2 );
      TESTER_ASSERT( t2( 2 ) == 3 );
      TESTER_ASSERT( t2( 3 ) == 4 );
   }

   void test_small_vector_bigger()
   {
      using V = Vector<int, AllocatorSingleStaticMemory<int, 4>>;

      V t1( 5 );
      t1 = { 1, 2, 3, 4, 5 };

      V t2 = t1;
      TESTER_ASSERT( &t1( 0 ) != &t2( 0 ) );
      TESTER_ASSERT( t2( 0 ) == 1 );
      TESTER_ASSERT( t2( 1 ) == 2 );
      TESTER_ASSERT( t2( 2 ) == 3 );
      TESTER_ASSERT( t2( 3 ) == 4 );
      TESTER_ASSERT( t2( 4 ) == 5 );
   }

   void test_small_normal()
   {
      using V = Vector<int, AllocatorSingleStaticMemory<int, 4>>;
      using V2 = Vector<int>;

      std::cout << "SIZEOFV=" << sizeof(V2) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory::value_type*) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory::value_type*) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory::index_mapper) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory::index_type) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(V2::Memory::allocator_type) << std::endl;
      std::cout << "SIZEOFV=" << sizeof(bool) << std::endl;

      V t1( 3 );
      t1 = { 1, 2, 3 };

      V t2( 3 );
      t2 = { 10, 20, 30 };

      const bool can_move = NAMESPACE_NLL::details::MemoryMoveable<V::Memory>::can_move(&t1(0), t1.size());
      TESTER_ASSERT(!can_move);

      auto t3 = t1 + t2;
      TESTER_ASSERT( t3( 0 ) == 11 );
      TESTER_ASSERT( t3( 1 ) == 22 );
      TESTER_ASSERT( t3( 2 ) == 33 );
   }

   void test_move_small()
   {
      using V = Vector<int, AllocatorSingleStaticMemory<int, 4>>;

      V t1(3);
      int* ptr_t1 = &t1(0);
      t1 = { 1, 2, 3 };

      V t2 = std::move(t1);
      TESTER_ASSERT(&t2(0) != ptr_t1);
      TESTER_ASSERT(t2(0) == 1);
      TESTER_ASSERT(t2(1) == 2);
      TESTER_ASSERT(t2(2) == 3);
   }

   void test_move_big()
   {
      using V = Vector<int, AllocatorSingleStaticMemory<int, 4>>;

      V t1(5);
      int* ptr_t1 = &t1(0);
      t1 = { 1, 2, 3, 4, 5 };

      V t2 = std::move(t1);
      TESTER_ASSERT(&t2(0) == ptr_t1);
      TESTER_ASSERT(t2(0) == 1);
      TESTER_ASSERT(t2(1) == 2);
      TESTER_ASSERT(t2(2) == 3);
      TESTER_ASSERT(t2(3) == 4);
      TESTER_ASSERT(t2(4) == 5);
   }
};

TESTER_TEST_SUITE( TestSmallArray );
TESTER_TEST( test_small_vector );
TESTER_TEST( test_small_vector_bigger );
TESTER_TEST( test_small_normal );
TESTER_TEST( test_move_small );
TESTER_TEST(test_move_big);
TESTER_TEST_SUITE_END();
