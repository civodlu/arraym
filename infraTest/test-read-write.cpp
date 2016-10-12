#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;


struct TestReadWrite
{
   // test default behavior for numeric types
   void testRWnumerics()
   {
      size_t   t1 = 128123;
      double   t2 = 4.331;
      char     t3 = 'u';
      bool     t4 = true;

      std::stringstream ss;
      core::write( t1, ss );
      core::write( t2, ss );
      core::write( t3, ss );
      core::write( t4, ss );

      core::read( t1, ss );
      core::read( t2, ss );
      core::read( t3, ss );
      core::read( t4, ss );

      TESTER_ASSERT( t1 == 128123 );
      TESTER_ASSERT( t2 == 4.331 );
      TESTER_ASSERT( t3 == 'u' );
      TESTER_ASSERT( t4 == true );

      TESTER_ASSERT( ss.good() );
   }

   // make sure that if the stream ran out we throw
   void testThrowIfStreamFailed()
   {
      bool hasThrow = false;

      try
      {
         std::stringstream ss;
         int n;
         core::read( n, ss );
      } catch (...)
      {
         hasThrow = true;
      }

      TESTER_ASSERT( hasThrow );
   }

   void testRWString()
   {
      const std::string s1 = "123456789";
      std::stringstream ss;

      std::string s2;
      core::write( s1, ss );
      core::read( s2, ss );
      TESTER_ASSERT( s2 == s1 );
   }

   void testRWSet()
   {
      std::stringstream ss;

      std::set<int> s1;
      s1.insert( 4 );
      s1.insert( 6 );
      s1.insert( 8 );

      std::set<int> s2;
      core::write( s1, ss );
      core::read( s2, ss );
      TESTER_ASSERT( s2 == s1 );
   }

   void testRWMap()
   {
      std::stringstream ss;

      std::map<int, std::string> s1;
      s1[ 4 ] = "s4";
      s1[ 6 ] = "s6";
      s1[ 8 ] = "s8";

      std::map<int, std::string> s2;
      core::write( s1, ss );
      core::read( s2, ss );
      TESTER_ASSERT( s2 == s1 );
   }
};

TESTER_TEST_SUITE(TestReadWrite);
 TESTER_TEST(testRWnumerics);
 TESTER_TEST(testThrowIfStreamFailed);
 TESTER_TEST(testRWString);
 TESTER_TEST(testRWSet);
 TESTER_TEST(testRWMap);
TESTER_TEST_SUITE_END();
