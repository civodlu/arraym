#pragma once

DECLARE_NAMESPACE_NLL

template <typename T, typename... Args>
Vector<T> make_vector( const Args&... args )
{
   const size_t size = sizeof...( Args );
   Vector<T> vector( size );
   if ( size )
   {
      T values[] = { static_cast<T>( args ) ... };
      std::copy_n( values, size, array_base_memory(vector) );
   }
   return vector;
}

template <typename T, typename... Args>
std::vector<T> make_stdvector( const Args&... args )
{
   const size_t size = sizeof...( Args );
   std::vector<T> vector( size );
   if ( size )
   {
      T values[] = { static_cast<T>( args ) ... };
      std::copy_n( values, size, vector.begin() );
   }
   return vector;
}

DECLARE_NAMESPACE_NLL_END