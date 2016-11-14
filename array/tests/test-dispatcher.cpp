#pragma warning(disable:4244)

#include <array/blas-dispatcher.h>
#include <tester/register.h>



using namespace NAMESPACE_NLL::blas;
using namespace NAMESPACE_NLL::blas::details;

struct TestBlasDispatcher
{
   static void saxpy_1( int, float, const float *, int, float *, int )
   {

   }

   static void daxpy_1( int, double, const double *, int, double *, int )
   {

   }

   void test_static()
   {
      BlasDispatcher::instance().get<BlasFunction::daxpy>();
   }

   void test_sxapy()
   {
      BlasDispatcherImpl dispatcher;
      dispatcher.registerFunction<BlasFunction::saxpy>( saxpy_1, "saxpy_1" );
      dispatcher.get<BlasFunction::saxpy>();

      dispatcher.registerFunction<BlasFunction::daxpy>( daxpy_1, "daxpy_1" );
      auto functions_daxpy = dispatcher.get<BlasFunction::daxpy>();
      TESTER_ASSERT( functions_daxpy.size() == 1 );

      {
         dispatcher.call<BlasFunction::saxpy>( float( 0 ), float( 0 ), (const float*)( nullptr ), int( 0 ), (float*)( nullptr ), int( 0 ) );
      }

      {
         // convertible arguments
         dispatcher.call<BlasFunction::saxpy>( double( 0 ), float( 0 ), (float*)( nullptr ), int( 0 ), (float*)( nullptr ), int( 0 ) );
      }
   }
};

TESTER_TEST_SUITE( TestBlasDispatcher );
TESTER_TEST( test_sxapy );
TESTER_TEST_SUITE_END();