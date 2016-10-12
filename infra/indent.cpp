#include "forward.h"

namespace nll
{
namespace core
{
   /// The slot to store the current indentation level.
   static long int indent_index = std::ios::xalloc();

   std::ostream& incindent( std::ostream& o )
   {
      #ifdef WITH_OMP
      #pragma omp critical
      #endif
      {
         o.iword(indent_index) += 2;
      }
      return o;
   }

   std::ostream& decindent( std::ostream& o )
   {
      ensure(o.iword( indent_index ), "indentation error" );
      #ifdef WITH_OMP
      #pragma omp critical
      #endif
      {
         o.iword( indent_index ) -= 2;
      }
      return o;
   }

   std::ostream& resetindent( std::ostream& o )
   {
      #ifdef WITH_OMP
      #pragma omp critical
      #endif
      {
         o.iword( indent_index ) = 0;
      }
      return o;
   }

   std::ostream& iendl( std::ostream& o )
   {
      o << std::endl;
      // Be sure to be able to restore the stream flags.
      char fill = o.fill(' ');

      #ifdef WITH_OMP
      #pragma omp critical
      #endif
      {
         o << std::setw( o.iword( indent_index ) );
      }
      o << "" << std::setfill( fill );
      return o;
   }

   std::ostream& incendl( std::ostream& o )
   {
      return o << incindent << iendl;
   }

   std::ostream& decendl( std::ostream& o )
   {
      return o << decindent << iendl;
   }
}
}
