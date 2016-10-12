#include "forward.h"

#if defined (_MSC_VER)   // windows version

#include <windows.h>
#include <psapi.h>

# pragma comment(lib, "Psapi.lib")

namespace nll
{
namespace core
{
   //
   // implementation based on http://stackoverflow.com/questions/63166/how-to-determine-cpu-and-memory-consumption-from-inside-a-process
   //
   float getVirtualMemoryUsedByThisProcess()
   {
      PROCESS_MEMORY_COUNTERS_EX pmc;
      const auto success = GetProcessMemoryInfo( GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS*)&pmc, sizeof(pmc) ); // tsss https://social.msdn.microsoft.com/Forums/en-US/720198c4-04a2-4737-9159-6e23a217d6b7/question-about-getprocessmemoryinfo?forum=Vsexpressvc
      if ( success )
      {
         // <PrivateUsage> is in bytes
         return (float)pmc.PrivateUsage / ( 1024 * 1024 );
      } else {
         return -1;
      }
   }
}
}

#else // linux version. Untested!

namespace nll
{
namespace core
{
   int parseLine( char* line )
   {
      int i = strlen(line);
      while (*line < '0' || *line > '9')
      {
         line++;
      }

      int end = std::max( 0, i - 3 );
      line[ end ] = '\0';
      i = atoi( line );
      return i;
   }

   float getVirtualMemoryUsedByThisProcess()
   {
      FILE* file = fopen("/proc/self/status", "r");
      if ( !file )
      {
         return -1;
      }

      float result = -1;
      char line[128];

      while ( fgets( line, 128, file ) != 0 )
      {
         if ( strncmp( line, "VmSize:", 7 ) == 0 )
         {
            result = parseLine( line );
            break;
         }
      }

      fclose(file);
      return result / 1024;
   }
}
}

#endif
