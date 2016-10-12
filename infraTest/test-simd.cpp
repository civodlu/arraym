#include <infra/forward.h>
#include <tester/register.h>

#include <boost/simd/sdk/simd/pack.hpp>
#include <boost/simd/sdk/simd/io.hpp>
#include <boost/simd/include/functions/sum.hpp>
#include <boost/simd/include/functions/splat.hpp>
#include <boost/simd/include/functions/plus.hpp>
#include <boost/simd/include/functions/multiplies.hpp>

using namespace nll;

#ifdef WITH_BOOSTSIMD

struct TestSimd
{
   void testSimd_basic()
   {
      typedef boost::simd::pack<float> p_t;

      p_t res;
      p_t u( 10 );
      p_t r = boost::simd::splat<p_t>( 11 );

      res = ( u + r ) * 2.f;
      std::cout << res << std::endl;

      TESTER_ASSERT( boost::simd::sum( res ) == 42 * 4 );
   }
};

TESTER_TEST_SUITE( TestSimd );
TESTER_TEST( testSimd_basic );
TESTER_TEST_SUITE_END();

#endif