#ifndef NLL_TESTER_CONFIG_H_
# define NLL_TESTER_CONFIG_H_

# include <map>
# include <iostream>
# include <fstream>
# include <vector>
# include <string>
# include <chrono>
# include <cstring>
# include <cstdlib>

namespace nll
{

   namespace impl
   {
      /**
      @ingroup core
      @brief Define a simple Timer class.
      */
      class Timer
      {
      public:
         /**
         @brief Instanciate the timer and start it
         */
         Timer()
         {
            start();
            _end = _start;
         }

         /**
         @brief restart the timer
         */
         void start()
         {
            _start = std::chrono::high_resolution_clock::now();
         }

         /**
         @brief end the timer, return the time in seconds spent
         */
         void end()
         {
            _end = std::chrono::high_resolution_clock::now();
         }

         /**
         @brief get the current time since the begining, return the time in seconds spent.
         */
         float getElapsedTime() const
         {
            auto c = std::chrono::high_resolution_clock::now();
            return toSeconds( _start, c );
         }

         /**
         @brief return the time in seconds spent since between starting and ending the timer. The timer needs to be ended before calling it.
         */
         float getTime() const
         {
            return toSeconds( _start, _end );
         }

      private:
         static float toSeconds( const std::chrono::high_resolution_clock::time_point& start,
                                 const std::chrono::high_resolution_clock::time_point& end )
         {
            const auto c = std::chrono::duration_cast<std::chrono::milliseconds>( end - start ).count();
            return static_cast<float>( c / 1000.0f );
         }

      private:
         std::chrono::high_resolution_clock::time_point _start;
         std::chrono::high_resolution_clock::time_point _end;
      };

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
               if ( !removeEmptyString || ( removeEmptyString && strlen( &str[ last ] ) ) )
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
   }

namespace tester
{
   

   /**
    @brief Save/Load/Modify a simple configuration file for regression testing
    */
   class Config
   {
   public:
      typedef std::map<std::string, std::string>   Directory;
      typedef std::map<std::string, Directory>     Storage;

   public:
      Config()
      {
      }

      Config( std::istream& i )
      {
         read( i );
      }

      Config( const std::string& i )
      {
         read( i );
      }

      const std::string& getDirectory() const
      {
         return _dir;
      }

      void setDirectory( const std::string& dir )
      {
         _dir = dir;
      }

      std::string& operator[]( const std::string& s )
      {
         return _storage[ _dir ][ s ];
      }

      void write( const std::string& file ) const
      {
         std::ofstream f( file.c_str() );
         if ( !f.is_open() )
            return;
         write( f );
      }

      void write( std::ostream& o ) const
      {
         o << "# This file has been automatically generated and should not be changed" << std::endl << std::endl;
         for ( Storage::const_iterator it = _storage.begin(); it != _storage.end(); ++it )
         {
            o << "[" << it->first << "]" << std::endl;
            for ( Directory::const_iterator it2 = it->second.begin(); it2 != it->second.end(); ++it2 )
            {
               o << it2->first << "=" << it2->second << std::endl;
            }
            o << std::endl;
         }
      }

      void read( const std::string& file )
      {
         std::ifstream f( file.c_str() );
         if ( !f.is_open() )
            return;
         read( f );
      }

      void read( std::istream& i )
      {
         _storage.clear();
         std::string line;
         std::string directory = "empty";

         std::getline( i, line );
         while ( !i.eof() )
         {
            if ( line.size() && line[ 0 ] != '#' )
            {
               size_t posStart = line.find_first_of( '[' );
               size_t posEnd = line.find_last_of( ']' );
               if ( posStart != std::string::npos && posEnd != std::string::npos )
               {
                  directory = std::string( &line[ posStart + 1 ], &line[ posEnd ] );
               } else {
                  std::vector<const char*> args = nll::impl::split( line, '=' );
                  if ( args.size() == 2 )
                  {
                     _storage[ directory ][ args[ 0 ] ] = args[ 1 ];
                  }
               }
            }
            std::getline( i, line );
         }
      }

   public:
      Storage      _storage;

   private:
      std::string  _dir;
   };
}
}

#endif
