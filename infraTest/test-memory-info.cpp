#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;

struct TestMemoryInfo
{
   void testDiffMem()
   {
      const float currentMem = core::getVirtualMemoryUsedByThisProcess();

      const float expectedMoAllocated = 40.15f;
      std::unique_ptr<char> n( new char[ (size_t)expectedMoAllocated * 1024 * 1024 ] );

      const float currentMemAfterAllocation = core::getVirtualMemoryUsedByThisProcess();
      TESTER_ASSERT( core::equal<float>( currentMemAfterAllocation - currentMem, expectedMoAllocated, 1 ) );
   }
};

TESTER_TEST_SUITE(TestMemoryInfo);
 TESTER_TEST(testDiffMem);
TESTER_TEST_SUITE_END();
