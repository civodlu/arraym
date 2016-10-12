#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;

struct TestUtilityPure
{
   void testEqual()
   {
      TESTER_ASSERT( core::equal<float>( 1, 1.5f, 0.51f ) );
      TESTER_ASSERT( !core::equal<float>( 1, 1.5f, 0.49f ) );

      TESTER_ASSERT( core::equal<float>( 1.5f, 1, 0.51f ) );
      TESTER_ASSERT( !core::equal<float>( 1.5f, 1, 0.49f ) );
   }

   void testStr2Strw()
   {
      std::wstring wstr = core::stringTowstring( "abC1" );
      TESTER_ASSERT( wstr == L"abC1" );
   }
};

TESTER_TEST_SUITE(TestUtilityPure);
 TESTER_TEST(testEqual);
 TESTER_TEST(testStr2Strw);
TESTER_TEST_SUITE_END();
