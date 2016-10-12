#include <infra/forward.h>
#include <tester/register.h>

using namespace nll;

struct TestTimer
{
   void testTimergetElapsedTime()
   {
      const float accuracy = 0.05f; // must at least be accurate as clock()
      float times[] =
      { 0.01f, 0.1f, 0.5f, 0.6f, 1.1f };

      core::Timer global;
      for ( auto t : times )
      {
         core::Timer timer;
         auto c = clock();

         while ( ( (float)( clock() - c ) / CLOCKS_PER_SEC ) < t )
            ;

         const float time = timer.getElapsedTime();
         std::cout << "time=" << time << " EXPECTED=" << t << std::endl;
         TESTER_ASSERT( core::equal( time, t, accuracy ) );
      }

      const float totalTime = global.getElapsedTime();
      TESTER_ASSERT( core::equal( totalTime, std::accumulate( times, times + core::getStaticBufferSize( times ), 0.0f), accuracy ) );
      TESTER_ASSERT( core::equal( global.getTime(), 0.0f ) );// we must end() the timer to update the time
      global.end();
      const float totalTimeEnd = global.getTime();
      TESTER_ASSERT( core::equal( totalTimeEnd, totalTime, 0.01f ) );
   }
};

TESTER_TEST_SUITE( TestTimer );
TESTER_TEST( testTimergetElapsedTime );
TESTER_TEST_SUITE_END();
