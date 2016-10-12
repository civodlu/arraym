#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;

#ifdef WITH_CUDA

struct TestCuda
{
   void testDummy()
   {
      const auto str = cuda::testCuda();
      std::cout << "STR=" << str << std::endl;
      TESTER_ASSERT( str == "Hello World!" );
      //core::RuntimeError e( "HAHAH" );
      //core::Timer t1;
      //std::cout << "T1=" << t1.getElapsedTime() << std::endl;
   }
};

TESTER_TEST_SUITE(TestCuda);
 TESTER_TEST(testDummy);
TESTER_TEST_SUITE_END();

#endif