#include "forward.h"

#ifdef _MSC_VER
// windows specific
# include <Windows.h>
# include <process.h>
# include <dbghelp.h>
# pragma comment(lib, "Kernel32.lib")
# pragma comment(lib, "Dbghelp.lib")
#else
// using glibc
# include <stdio.h>

#ifdef NLL_HAS_EXECINFO
# include <execinfo.h>
#endif

# include <signal.h>
# include <stdlib.h>
# include <unistd.h>
#endif

namespace nll
{
namespace core
{
   RuntimeError::RuntimeError( const std::string& msg, bool captureStackTrace, bool displayStackTrace ) throw()
   {
      if ( captureStackTrace )
      {
         try
         {
            std::stringstream ss;
            ss << msg << std::endl
               << "------------------" << std::endl
               << "--- StackTrace ---" << std::endl
               << "------------------" << std::endl;

            std::stringstream trace;
            core::printCallStack( trace );
            if ( displayStackTrace )
            {
               ss << trace.str();
            }

            _msg = ss.str();
            _callstack = trace.str();
         } catch( ... )
         {
            // do nothing. Just catch any exception else this will terminate the process!
         }
      } else {
         _msg = msg;
      }
   }

   const char* RuntimeError::what() const throw()
   {
      return _msg.c_str();
   }

   const char* RuntimeError::trace() const throw()
   {
      return _callstack.c_str();
   }

   static const size_t nbStacksLevel = 200;

   #if defined( WIN32 ) || defined( WIN64 )
   /**
    Win32 & Win64 specific implementation
    */
   void printCallStack( std::ostream& o )
   {
      HANDLE process = GetCurrentProcess();
      SymInitialize( process, NULL, TRUE );     // RAAAAAHHHHH, it seems it NEVER returns the correct value (e.g., http://emptyhammock.blogspot.de/2012/08/gotta-love-syminitialize.html)
      //if ( success )
      {
         void* stack[ nbStacksLevel ];
         const unsigned short frames = CaptureStackBackTrace( 0, nbStacksLevel, stack, NULL );
         SYMBOL_INFO* symbol = (SYMBOL_INFO *)calloc( sizeof( SYMBOL_INFO ) + 256 * sizeof( char ), 1 );
         symbol->MaxNameLen   = 255;
         symbol->SizeOfStruct = sizeof( SYMBOL_INFO );

         #ifdef WIN64
         #  pragma message( "WIN64 RuntimeError::RuntimeError" )
         IMAGEHLP_LINE64 *line = (IMAGEHLP_LINE64*)malloc(sizeof(IMAGEHLP_LINE64));
         line->SizeOfStruct = sizeof(IMAGEHLP_LINE64);
         #else
         #  pragma message( "WIN32 RuntimeError::RuntimeError" )
         IMAGEHLP_LINE *line = (IMAGEHLP_LINE *)malloc(sizeof(IMAGEHLP_LINE));
         line->SizeOfStruct = sizeof(IMAGEHLP_LINE);
         #endif

         for( int i = 1; i < frames; i++ )   // frame 0 is this function
         {
            line->FileName = 0;
            symbol->Address = 0;
            line->LineNumber = 0;

            const BOOL success1 = SymFromAddr( process, ( DWORD64 )( stack[ i ] ), 0, symbol );

            #pragma warning(disable:4302)
            DWORD dwDisplacement = 0;
            #ifdef WIN64
            const BOOL success2 = SymGetLineFromAddr64(process, (DWORD64)(stack[i]), &dwDisplacement, line);
            #else
            const BOOL success2 = SymGetLineFromAddr(process, (DWORD)(stack[i]), &dwDisplacement, line);
            #endif

            if ( symbol->Name && line->FileName && success2 )
            {
               o << "[" << i << "] name=" << symbol->Name << " file=" << line->FileName << " line=" << line->LineNumber << std::endl;
            } else {
               if ( success1 )
               {
                  o << "[" << i << "] adress=" << symbol->Address << std::endl;
               } else {
                  o << "[" << i << "] adress=unknown" << std::endl;
               }
            }
         }

         free( line );
         free( symbol );
      } /*else {
         const DWORD error = GetLastError();
         std::cerr << "Failed to retrieve call stack!, SymInitialize error=" << error << std::endl;
      }*/
   }
   #else
   #ifdef NLL_HAS_EXECINFO
   // using glibc
   // see http://stackoverflow.com/questions/77005/how-to-generate-a-stacktrace-when-my-gcc-c-app-crashes
   void printCallStack( std::ostream& o )
   {
      void* array[ nbStacksLevel ];
      const size_t size = backtrace( array, nbStacksLevel );
      char** messages = backtrace_symbols( array, size );

      if ( messages )
      {
         for ( auto i = 1; i < size; ++i )
         {
             o << "[" << i << "] name=" << messages[i] << std::endl;
         }
         free( messages );
      }
   }
   #else
      void printCallStack( std::ostream& o )
      {
         // do nothing. We can't have access to the frame info
      }
   #endif
   #endif
}
}
