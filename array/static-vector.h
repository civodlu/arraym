/*
* Numerical learning library
* http://nll.googlecode.com/
*
* Copyright (c) 2009-2015, Ludovic Sibille
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

#ifndef NLL_STATIC_VECTOR_H_
# define NLL_STATIC_VECTOR_H_


namespace nll
{
   namespace core
   {
      template <class T, int SIZE>
      class StaticVector;
   }
}

namespace nll
{
   namespace core
   {
      struct No_init_tag
      {};

      const static No_init_tag no_init_tag;

      /**
      @ingroup core
      @brief implement static vectors. Memory is allocated on the heap. Memory is not shared accross instances.
      */
      template <class T, int SIZE>
      class StaticVector
      {
      public:
         typedef T            value_type;
         typedef T*           iterator;
         typedef const T*     const_iterator;

         typedef T*           pointer;
         typedef const T*     const_pointer;
         typedef T&           reference;
         typedef const T&     const_reference;

         enum
         {
            sizeDef = SIZE
         };

         static_assert( std::is_trivially_copyable<T>::value, "must be trivial to copy" );

      public:
         /**
         @brief copy constructor
         */
         StaticVector( const StaticVector& cpy )
         {
            memcpy( begin(), cpy.begin(), SIZE * sizeof( T ) );
         }

         /**
         @brief instantiate a vector
         */
         StaticVector()
         {
            std::fill_n( begin(), SIZE, T() );
         }

         explicit StaticVector( const No_init_tag& UNUSED( noInit ) )
         {
            // do nothing
         }

         template <class T2>
         explicit StaticVector( const StaticVector<T2, SIZE>& cpy )
         {
            for ( size_t n = 0; n < SIZE; ++n )
            {
               begin()[ n ] = static_cast<T>( cpy[ n ] );
            }
         }

         
         template <typename T2, typename... Args, typename = typename std::enable_if<sizeof...(Args) + 1 == SIZE>::type /*, typename = typename std::enable_if<core::is_same<Args...>::value>::type*/>
         StaticVector( const T2& arg1, const Args&... args ) // avoid "shadowing" the default constructor with an initia argument
         {
            begin()[ 0 ] = static_cast<T>( arg1 );
            init<1>( args... );
         }

         StaticVector( const T& value )
         {
            std::fill_n( begin(), SIZE, value );
         }

      private:
         template <int Dim, typename T2, typename... Args>
         FORCE_INLINE void init( const T2& value, const Args&... args )
         {
            begin()[ Dim ] = static_cast<T>( value );
            init<Dim + 1>( args... );
         }

         // end recursion
         template <int Dim>
         FORCE_INLINE void init()
         {}


      public:
         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE const T& at( const size_t index ) const
         {
            NLL_FAST_ASSERT( index < SIZE, "out of bound index" );
            return _buffer[ index ];
         }

         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE T& at( const size_t index )
         {
            NLL_FAST_ASSERT( index < SIZE, "out of bound index" );
            return _buffer[ index ];
         }

         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE const T& operator[]( const size_t index ) const
         {
            return at( index );
         }

         /**
         @brief return the value at the specified index
         */
         template <int N>
         FORCE_INLINE const T& get() const
         {
            static_assert( N < SIZE, "N must be < SIZE" );
            return _buffer[ N ];
         }

         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE T& operator[]( const size_t index )
         {
            return at( index );
         }

         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE const T& operator()( const size_t index ) const
         {
            return at( index );
         }

         /**
         @brief return the value at the specified index
         */
         FORCE_INLINE T& operator()( const size_t index )
         {
            return at( index );
         }

         template <class T2>
         StaticVector<T2, SIZE> staticCastTo() const
         {
            StaticVector<T2, SIZE> casted;
            for ( size_t n = 0; n < SIZE; ++n )
            {
               casted[ n ] = static_cast<T2>( _buffer[ n ] );
            }
            return casted;
         }

         FORCE_INLINE static size_t size()
         {
            return SIZE;
         }

         /**
         @brief print the vector to a stream
         */
         void print( std::ostream& o ) const
         {
            o << "[";
            if ( SIZE )
            {
               for ( size_t n = 0; n + 1 < SIZE; ++n )
                  o << at( n ) << " ";
               o << at( SIZE - 1 );
            }
            o << "]";
         }

         void write( std::ostream& f ) const
         {
            f.write( (i8*)_buffer, sizeof( T ) * SIZE );
         }

         void read( std::istream& f )
         {
            f.read( (i8*)_buffer, sizeof( T ) * SIZE );
         }

         FORCE_INLINE bool equal( const StaticVector& r, T eps = ( T )1e-5 ) const
         {
            for ( size_t n = 0; n < SIZE; ++n )
            {
               if ( !core::equal( _buffer[ n ], r[ n ], eps ) )
                  return false;
            }
            return true;
         }

         FORCE_INLINE bool operator==( const StaticVector& r ) const
         {
            return equal( r );
         }

         FORCE_INLINE bool operator!=( const StaticVector& r ) const
         {
            return !( *this == r );
         }

         FORCE_INLINE iterator begin()
         {
            return _buffer;
         }

         FORCE_INLINE const_iterator begin() const
         {
            return _buffer;
         }

         FORCE_INLINE iterator end()
         {
            return _buffer + SIZE;
         }

         FORCE_INLINE const_iterator end() const
         {
            return _buffer + SIZE;
         }

         /*
         StaticVector<T, SIZE>& operator=( const T& value )
         {
            std::fill_n( begin(), SIZE, value );
            return *this;
         }*/

      protected:
         T     _buffer[ SIZE ];
      };

      /**
      @ingroup core
      @brief equal operator
      */
      template <class T, int SIZE>
      bool operator==( const StaticVector<T, SIZE>& l, const StaticVector<T, SIZE>& r )
      {
         for ( size_t n = 0; n < SIZE; ++n )
         {
            if ( !equal( l[ n ], r[ n ] ) )
               return false;
         }
         return true;
      }

      template <class T, int Size>
      bool equal( const StaticVector<T, Size>& lhs, const StaticVector<T, Size>& rhs, T eps = ( T )1e-5 )
      {
         return lhs.equal( rhs, eps );
      }

      template <class T, int SIZE>
      std::ostream& operator<<( std::ostream& o, const StaticVector<T, SIZE>& v )
      {
         v.print( o );
         return o;
      }

      using vector1ui = StaticVector<ui32, 1>;
      

      using vector2ui = StaticVector<ui32, 2>;
      using vector2i = StaticVector<i32, 2>;
      using vector2d  = StaticVector<f64, 2>;

      using vector3ui = StaticVector<ui32, 3>;
      
      using vector1uc = StaticVector<ui8, 1>;
      using vector2uc = StaticVector<ui8, 2>;
      using vector3uc = StaticVector<ui8, 3>;
      using vector4uc = StaticVector<ui8, 4>;

      using vector1c = StaticVector<i8, 1>;
      using vector2c = StaticVector<i8, 2>;
      using vector3c = StaticVector<i8, 3>;
      using vector4c = StaticVector<i8, 4>;

      using vector1f = StaticVector<f32, 1>;
      using vector2f = StaticVector<f32, 2>;
      using vector3f = StaticVector<f32, 3>;
      using vector4f = StaticVector<f32, 4>;

      using vector3i = StaticVector<i32, 3>;
      using vector4f = StaticVector<f32, 4>;
      using vector3d = StaticVector<f64, 3>;
      using vector3int = StaticVector<int, 3>;
      using vector3uint = StaticVector<size_t, 3>;
   }
}

#endif
