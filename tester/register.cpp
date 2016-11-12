#include "register.h"
#include <iostream>

// can be compiled without openmp
#ifndef NO_MULTITHREADED_UNIT_TESTS
#include <omp.h>
#endif

#include <assert.h>


unsigned Register::run( bool canBeMultithreaded )
{
   if ( canBeMultithreaded )
   {
      #ifndef NO_MULTITHREADED_UNIT_TESTS
      #pragma omp parallel for
      #endif
      for ( int n = 0; n < static_cast<int>( _suites.size() ); ++ n )
      {
         _suites[ n ]->run();
      }
   } else {
      for ( int n = 0; n < static_cast<int>( _suites.size() ); ++ n )
      {
         _suites[ n ]->run();
      }
   }

   TESTER_STREAM << std::endl;
   if ( !_faileds.size() )
   {
      TESTER_STREAM << "success (" << _successful << " tests)" << std::endl << std::endl;
      regressionExport();
      return 0;
   }

   TESTER_STREAM << "failure (success:" << _successful << ", failures:" << _faileds.size() << ")" << std::endl << std::endl;
   TESTER_STREAM << "test failed:" << std::endl;
   for ( unsigned n = 0; n < _faileds.size(); ++n )
   {
      TESTER_STREAM << "  " << _faileds[ n ].file << ": " << _faileds[ n ].funcName << ": " << _faileds[ n ].msg << std::endl;
   }

   return static_cast<unsigned>( _faileds.size() );
}
