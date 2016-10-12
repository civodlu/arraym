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

#ifndef NLL_IO_H_
# define NLL_IO_H_

namespace nll
{
namespace core
{
   template <class T> void write( const T& val, std::ostream& f );
   template <class T> void read( T& val, std::istream& f );

   template <class T, bool isNative>
   struct _write
   {
      _write( const T& val, std::ostream& f )
      {
         val.write( f );
      }
   };

   template <class T>
   struct _write<T, true>
   {
      _write( const T& val, std::ostream& f )
      {
         f.write( (i8*)( &val ), sizeof ( val ) );
      }
   };

   template <>
   struct _write<bool, true>
   {
      _write( const bool& val, std::ostream& f )
      {
         i8 value = val;
         f.write( (i8*)( &value ), sizeof ( value ) );
      }
   };

   template <>
   struct _write<size_t, true>
   {
      _write( const size_t& val, std::ostream& f )
      {
         unsigned long long value = val;
         f.write( (i8*)( &value ), sizeof ( value ) );
      }
   };

   template <class T>
   struct _write<std::vector<T>, false>
   {
      _write( const std::vector<T>& val, std::ostream& f )
      {
         ui32 size = static_cast<ui32>( val.size() );
         write<ui32>( size, f );
         for ( ui32 n = 0; n < size; ++n )
            write<T>( val[ n ], f );
      }
   };

   template <>
   struct _write<std::vector<bool>, false>
   {
      _write( const std::vector<bool>& val, std::ostream& f )
      {
         ui32 size = static_cast<ui32>( val.size() );
         write<ui32>( size, f );
         for ( ui32 n = 0; n < size; ++n )
         {
            unsigned char value = static_cast<unsigned char>( val[ n ] );
            write( value, f );
         }
      }
   };

   template <class T1, class T2>
   struct _write<std::map<T1, T2>, false>
   {
      _write( const std::map<T1, T2>& val, std::ostream& f )
      {
         ui32 size = static_cast<ui32>( val.size() );
         write( size, f );

         for ( auto it = val.begin(); it != val.end(); ++it )
         {
            write( it->first, f );
            write( it->second, f );
         }
      }
   };

   template <class T1>
   struct _write<std::set<T1>, false>
   {
      _write( const std::set<T1>& val, std::ostream& f )
      {
         ui32 size = static_cast<ui32>( val.size() );
         write( size, f );

         for ( auto it = val.begin(); it != val.end(); ++it )
         {
            write( *it, f );
         }
      }
   };

   /**
    @ingroup core
    @brief write data to a stream. If native type, write it using stream functions, else the type needs to provide write()
    */
   template <class T> void write( const T& val, std::ostream& f )
   {
      _write<T, std::is_trivially_copyable<T>::value || std::is_scalar<T>::value>( val, f );
      ensure( f.good(), "stream error!" );
   }

   template <class T, bool isNative>
   struct _read
   {
      _read( T& val, std::istream& f)
      {
         val.read( f );
      }
   };

   template <class T>
   struct _read<T, true>
   {
      _read( T& val, std::istream& f)
      {
         f.read( (i8*)( &val ), sizeof ( val ) );
      }
   };

   template <>
   struct _read<bool, true>
   {
      _read( bool& val, std::istream& f)
      {
         i8 value = 0;
         f.read( (i8*)( &value ), sizeof ( value ) );
         val = ( value != 0 );
      }
   };

   template <>
   struct _read<size_t, true>
   {
      _read( size_t& val, std::istream& f)
      {
         unsigned long long value = 0;
         f.read( (i8*)( &value ), sizeof ( value ) );
         val = value;
      }
   };

   template <class T>
   struct _read<std::vector<T>, false>
   {
      _read( std::vector<T>& val, std::istream& i )
      {
         ui32 size = 0;
         read<ui32>( size, i );
         val = std::vector<T>( size );
         for ( ui32 n = 0; n < size; ++n )
            read<T>( val[ n ], i );
      }
   };

   template <>
   struct _read<std::vector<bool>, false>
   {
      _read( std::vector<bool>& val, std::istream& i )
      {
         ui32 size = 0;
         read<ui32>( size, i );
         val = std::vector<bool>( size );
         for ( ui32 n = 0; n < size; ++n )
         {
            unsigned char value = 0;
            read( value, i );
            val[ n ] = ( value != 0 );
         }
      }
   };

   template <class T1, class T2>
   struct _read<std::map<T1, T2>, false>
   {
      _read( std::map<T1, T2>& val, std::istream& f )
      {
         ui32 size = 0;
         read<ui32>( size, f );
         for ( ui32 s = 0; s < size; ++s )
         {
            T1 first;
            T2 second;

            read( first, f );
            read( second, f );
            val.insert( std::make_pair( first, second ) );
         }
      }
   };

   template <class T1>
   struct _read<std::set<T1>, false>
   {
      _read( std::set<T1>& val, std::istream& f )
      {
         ui32 size = 0;
         read<ui32>( size, f );
         for ( ui32 s = 0; s < size; ++s )
         {
            T1 first;

            read( first, f );
            val.insert( first );
         }
      }
   };

   template <>
   struct _write<std::string, false>
   {
      _write( const std::string& val, std::ostream& f )
      {
         ui32 size = static_cast<ui32>( val.size() );
         write<ui32>( size, f );
         f.write( val.c_str(), size );
      }
   };

   template <>
   struct _read<std::string, false>
   {
      _read( std::string& val, std::istream& i )
      {
         ui32 size = 0;
         read<ui32>( size, i );
         val = std::string( size, ' ' );
         if ( size )
         {
            i.read( &val[ 0 ], size );
         }
      }
   };

   /**
    @ingroup core
    @brief write data to a stream. If native type, write it using stream functions, else the type needs to provide write()
    */
   template <class T> void read( T& val, std::istream& f )
   {
      _read<T, std::is_trivially_copyable<T>::value || std::is_scalar<T>::value>( val, f );
      ensure( f.good(), "stream error!" );
   }
}
}

#endif
