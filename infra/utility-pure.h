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

#ifndef NLL_UTILITY_PURE_H_
# define NLL_UTILITY_PURE_H_

# ifdef NLL_BOUND
#  undef NLL_BOUND
# endif

/**
 @brief bound a value between 2 bounds. Special warning: min and max type have to match val type!!
 */
# define NLL_BOUND(val, min, max) ( ( ( val ) > ( max ) ) ? ( max ) : ( ( ( val ) < ( min ) ) ? ( min ) : ( val ) ) )


//
// define a IS_NAN macro
//
# ifdef IS_NAN
#  undef IS_NAN
# endif
# ifdef _MSC_VER
#  define IS_NAN( x )    ( _isnan( x ) )
# else
#  define IS_NAN( x )    ( std::isnan( x ) )
# endif

namespace nll
{
namespace core
{
   /**
    @ingroup core
    @brief Forbids assignment
    */
   class INFRA_API NonAssignable
   {
   public:
      NonAssignable()
      {}

   private:
      NonAssignable& operator=( const NonAssignable& );
   };

   /**
    @ingroup core
    @brief Inherit from this class to make explicitly a class not copyiable
    */
   struct INFRA_API NonCopyable
   {
      NonCopyable()
      {}

   private:
      NonCopyable & operator=(const NonCopyable&);
      NonCopyable(const NonCopyable&);
      NonCopyable(NonCopyable&&);
   };

   /**
    @ingroup core
    @brief Generic factory generator with no parameters
    */
   template <class Object>
   class FactoryGeneric
   {
   public:
      typedef Object value_type;

      static value_type create()
      {
         return value_type();
      }
   };

   template <class Object>
   class FactoryGenericSharedPtr
   {
   public:
      typedef Object value_type;

      static std::shared_ptr<value_type> create()
      {
         return std::shared_ptr<value_type>( new value_type() );
      }
   };

   /**
    @ingroup core
    @brief calculate an absolute value. Work around for Visual and ambiguous case...
    */
   inline double absolute( double val )
   {
      return val >= 0 ? val : - val;
   }

   template <class T>
   inline T absoluteT( T val )
   {
      return val >= 0 ? val : - val;
   }

   /**
    @ingroup core
    @brief test if 2 values are equal with a certain tolerance
   */
   template <class T, typename = typename std::enable_if< std::is_arithmetic<T>::value >::type>
   bool equal( const T val1, const T val2, const T tolerance = 2 * std::numeric_limits<T>::epsilon() )
   {
      return absolute( (double)val1 - (double)val2 ) <= (double)tolerance;
   }

   /**
   @ingroup core
   @brief test if val1 <= val2 + tolerance
   @return true if val1 <= val2 + tolerance
   */
   template <class T>
   bool less( const T val1, const T val2, const T tolerance = 2 * std::numeric_limits<T>::epsilon() )
   {
      return (double)val1 <= ( (double)val2 + (double)tolerance );
   }

   /**
    @ingroup core
    @brief compute statically the size of an array defined like this: T array[] = { .... }
   */
	template <std::size_t N, class T> size_t getStaticBufferSize(const T (& /*Elements*/)[N])
	{
		return N;
	}

   /**
    @ingroup core
    @brief compute a string representing this value
   */
   template <class T>
   std::string val2str( const T& val )
   {
      std::stringstream s;
      s << val;
      std::string res;
      s >> res;

      ensure( !s.fail(), "conversion failed for value=" << val );
      return res;
   }

   /**
    @ingroup core
    @brief compute a string representing this value
   */
   template <class T>
   std::string val2strHex( const T& val )
   {
      std::stringstream s;
      s << std::hex << val;
      std::string res;
      s >> res;

      ensure( !s.fail(), "conversion failed for value=" << val );
      return res;
   }

   /**
    @ingroup core
    @brief compute a value representing a string
   */
   template <class T>
   T str2val( const std::string& val )
   {
      std::stringstream s;
      s << val;
      T res;
      s >> res;

      ensure( !s.fail(), "conversion failed for value=" << val );
      return res;
   }

   /**
    @ingroup core
    @brief split a string according to a specific separator.
    
     The input string is altered: each time the separator is found, it is replaced by a null character.
     Each entry of the returned vector point to a part of the string. All empty strings are removed.
    */
   inline std::vector<const char*> split( std::string& str, char separator = ' ', bool removeEmptyString = true )
   {
      std::vector<const char*> s;
      unsigned last = 0;
      for ( unsigned n = 0; n < str.size(); ++n )
      {
         if ( str[ n ] == separator )
         {
            str[ n ] = 0;
            if ( !removeEmptyString || ( removeEmptyString && std::strlen( &str[ last ] ) ) )
               s.push_back( &str[ last ] );
            last = n + 1;
         }
      }

      if ( last < str.size() )
      {
         s.push_back( &str[ last ] );
      }
      return s;
   }

   /**
    @ingroup core
    @brief convert a string to a wstring
    */
   inline std::wstring stringTowstring( const std::string& s )
   {
      std::wstring temp;
      temp.assign( s.begin(), s.end() );
      if ( temp.size() )
      {
         temp[ 0 ] = static_cast<std::wstring::value_type>( s[ 0 ] );
      }
      return temp; 
   }
}
}

#endif
