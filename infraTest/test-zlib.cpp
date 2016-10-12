#include <array/forward.h>
#include <tester/register.h>

using namespace nll;

#ifdef WITH_ZLIB
struct TestArrayCompression
{
   void testCompressGzip()
   {
      gzofstream f( PATH_TO_TEST_DATA "tmp/f1.gzip" );
      TESTER_ASSERT( f.good() );

      srand( 0 );
      std::array<ui8, 1024> array;
      for ( auto& v : array )
      {
         v = rand() % 256;
      }
      f.write( reinterpret_cast<char*>( &array[ 0 ] ), array.size() );
      f.close();

      gzifstream f2( PATH_TO_TEST_DATA "tmp/f1.gzip" );
      TESTER_ASSERT( f2.good() );
      std::array<ui8, 1024> array_read;
      f2.read( reinterpret_cast<char*>( &array_read[ 0 ] ), array.size() );
      for ( size_t n = 0; n < array_read.size(); ++n )
      {
         TESTER_ASSERT( array[ n ] == array_read[ n ] );
      }
   }
};

TESTER_TEST_SUITE( TestArrayCompression );
TESTER_TEST( testCompressGzip );
TESTER_TEST_SUITE_END();


#endif