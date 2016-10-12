/*
 * Numerical learning library
 * http://nll.googlecode.com/
 *
 * Copyright (c) 2009-2012, Ludovic Sibille
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Ludovic Sibille nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY LUDOVIC SIBILLE ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE REGENTS AND CONTRIBUTORS BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef NLL_CORE_SYMBOL_H_
# define NLL_CORE_SYMBOL_H_

namespace nll
{
namespace core
{
   /**
    @ingroup core
    @brief Class representing a lightweight string. Only one instance for each string can be created,
           so we can compare directly string internal buffer pointer for comparison

    The cost at creation time relatively high, however once the symbol is created it is extremely fast
    */
   class INFRA_API Symbol
   {
   public:
      typedef std::set<std::string> Strings;
      friend class SymbolHolder;

      /**
       @brief Create a symbol.
       @note this symbol is stored in a static map!
       @see <SymbolHolder> for a independent holder
       */
      static Symbol create( const std::string& s )
      {
         // return the address contained in the set, guaranteing its unicity
         std::pair<Strings::iterator, bool> it;

         #ifdef WITH_OMP
         #pragma omp critical
         #endif
         {
            // protect against different threads accessing the shared strings
            it = _strings.insert( s );
         }
         return Symbol( it.first->c_str() );
      }

      void write( std::ostream& f ) const
      {
         const std::string s = _s;
         core::write( s, f );
      }

      void read( std::istream& f )
      {
         std::string s;
         core::read( s, f );
         _s = create( s )._s;
      }

      /**
       @brief Create a symbol.
       @note this symbol is stored in a static map!
       @see <SymbolHolder> for a independent holder
       */
      static Symbol create( const char* s )
      {
         // return the address contained in the set, guaranteing its unicity
         std::pair<Strings::iterator, bool> it;

         #ifdef WITH_OMP
         #pragma omp critical
         #endif
         {
            // protect against different threads accessing the shared strings
            it = _strings.insert( std::string( s ) );
         }
         return Symbol( it.first->c_str() );
      }

      bool operator==( const Symbol& rhs ) const
      {
         return _s == rhs._s;
      }

      bool operator!=( const Symbol& rhs ) const
      {
         return _s != rhs._s;
      }

      bool operator<( const Symbol& rhs ) const
      {
         return _s < rhs._s;
      }

      const char* getName() const
      {
         return _s;
      }

   public:
      Symbol() : _s( 0 )
      {
      }

   protected:
      // to be created internally only!
      Symbol( const char* s ) : _s( s )
      {
      }

   protected:
      static Strings _strings;
      const char* _s;
   };

   inline std::ostream&
   operator<<( std::ostream& ostr, const Symbol& s )
   {
      // TODO : hmm ostr << s.getValue() doesn't work with CLion??? strange...
      std::ostream* ostrp = &ostr;
      const char* value = s.getName();
      *ostrp << value;
      return ostr;
   }

   /**
    @ingroup core
    @brief Use an independent holder

    Symbols are only valid as long as the holder exist
    */
   class INFRA_API SymbolHolder
   {
      typedef std::set<std::string> Strings;

   public:
      Symbol create( const std::string& s )
      {
         // return the address contained in the set, guaranteing its unicity
         const std::pair<Strings::iterator, bool> it = _strings.insert( s );
         return Symbol( it.first->c_str() );
      }

      Symbol create( const char* s )
      {
         std::pair<Strings::iterator, bool> it;
         it = _strings.insert( std::string( s ) );
         return Symbol( it.first->c_str() );
      }

   private:
      Strings _strings;
   };
}
}

#endif
