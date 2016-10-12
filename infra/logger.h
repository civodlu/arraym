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

#ifndef NLL_LOGGER_H_
# define NLL_LOGGER_H_

namespace nll
{
namespace core
{
   class INFRA_API LoggerFormatter : public core::NonCopyable
   {
   public:
      virtual std::string format( const std::string& msg, int loglevel ) const = 0;
   };

   class INFRA_API LoggerFormatterDefault : public LoggerFormatter
   {
   public:
      virtual std::string format( const std::string& msg, int loglevel ) const override
      {
         #ifdef NLL_HAS_OMP_H
         const int threadId = omp_get_thread_num();
         #else
         const int threadId = 0;
         #endif

         static const char* stringsLogLevel[] =
         {
            "impl",
            "info",
            "warning",
            "error"
         };
         assert( loglevel < 4 );

         std::stringstream ss;
         ss << "[" << stringsLogLevel[ loglevel ] << "][" << threadId << "] " << msg << ", t=" << _time.getElapsedTime();
         return ss.str();
      }

   private:
      core::Timer _time;
   };

   class INFRA_API Logger : public core::NonCopyable
   {
   public:
      typedef std::ostream& (*fun_pretty_print_ptr)( std::ostream& o );

      enum LogLevel
      {
         LOG_IMPLEMENTATION,
         LOG_INFO,
         LOG_WARNING,
         LOG_ERROR
      };

      struct EnterFunction
      {
         EnterFunction( Logger* logger, const std::string& name, LogLevel logLevelToPrint ) : _logger( logger ), _name( name ), _logLevelToPrint( logLevelToPrint )
         {
            if ( _logger->getLogLevel() <= _logLevelToPrint )
            {
               std::stringstream ss;
               ss << "+" << _name;
               _logger->writeln( ss.str(), _logLevelToPrint );
            }
            _logger->write( core::incindent );
         }

         ~EnterFunction()
         {
            if ( !_logger )
               return;

            _logger->write( core::decindent );
            if ( _logger->getLogLevel() <= _logLevelToPrint )
            {
               std::stringstream ss;
               ss << "-" << _name;
               _logger->writeln( ss.str(), _logLevelToPrint );
            }
         }

         void setLogLevel( LogLevel logLevelToPrint )
         {
            _logLevelToPrint = logLevelToPrint;
         }

      private:
         Logger*        _logger;
         std::string    _name;
         LogLevel       _logLevelToPrint;
      };

      Logger( const std::shared_ptr<std::ostream>& stream, const std::shared_ptr<LoggerFormatter>& formatter, LogLevel logLevel = LOG_INFO ) : _stream( stream ), _formatter( formatter ), _logLevel( logLevel )
      {}

      void writeln( const std::string& msg, LogLevel logLevel )
      {
         if ( logLevel >= _logLevel )
         {
            (*_stream) << core::iendl << _formatter->format( msg, (int)logLevel );
         }
      }

      void writeln( std::stringstream& msg, LogLevel logLevel )
      {
         if ( logLevel >= _logLevel )
         {
            while ( !msg.eof() )
            {
               std::string line;
               std::getline( msg, line );
               if ( line.size() )
               {
                  (*_stream) << core::iendl << _formatter->format( line, (int)logLevel );
               }
            }
         }
      }

      void write( fun_pretty_print_ptr fn )
      {
         (*_stream) << fn;
      }

      LogLevel getLogLevel() const
      {
         return _logLevel;
      }

      void setLogLevel( LogLevel newLevel )
      {
         _logLevel = newLevel;
      }

   protected:
      std::shared_ptr<std::ostream>       _stream;
      std::shared_ptr<LoggerFormatter>    _formatter;
      LogLevel                            _logLevel;
   };

   class INFRA_API _LoggerNll : public Logger
   {
   public:
      _LoggerNll() : Logger( std::shared_ptr<std::ostream>( new std::ofstream( "nll.log" ) ),
                             std::shared_ptr<LoggerFormatter>( new LoggerFormatterDefault() ),
                             LOG_IMPLEMENTATION )
      {
         ensure( _stream->good(), "looger stream could not be opened!" );
      }
   };

   // explicitly export the <LoggerNll> template specialization so that the internal state is not lost in
   // the other binaries (i.e., from another shared lib, the template would be re-created and logging lost)
   template class INFRA_API core::Singleton<_LoggerNll>;


   typedef core::Singleton<_LoggerNll> LoggerNll;

   /**
   @brief temporarily modify the level of the logging. When destructed, this object restored the initial log level
   */
   class LogLevelModifierTemporary : public core::NonCopyable
   {
   public:
      LogLevelModifierTemporary( Logger::LogLevel newLevel )
      {
         _previous = LoggerNll::instance().getLogLevel();
         LoggerNll::instance().setLogLevel( newLevel );
      }

      ~LogLevelModifierTemporary()
      {
         LoggerNll::instance().setLogLevel( _previous );
      }

   private:
      Logger::LogLevel _previous;
   };

   //
   // The following macro can be redefined to allow NLL to integrate with different logging libraries (or completely disabled)
   //

   // if this causes a redefinition, it means this macro was called several times inside this function!
   #ifndef NLL_LOG_TRACE
   # define NLL_LOG_TRACE()  nll::core::Logger::EnterFunction functionEnteredLock( &nll::core::LoggerNll::instance(), __func__, nll::core::Logger::LOG_IMPLEMENTATION )
   #endif

   #ifndef NLL_LOG_TRACE_NAMED
   # define NLL_LOG_TRACE_NAMED(name) nll::core::Logger::EnterFunction functionEnteredLock( &nll::core::LoggerNll::instance(), name, nll::core::Logger::LOG_IMPLEMENTATION )
   #endif

   #ifndef NLL_LOG_IMPL
   #define NLL_LOG_IMPL(msg)                                            \
   {                                                                    \
      auto& _logPrivate = nll::core::LoggerNll::instance();                     \
      if ( _logPrivate.getLogLevel() <= nll::core::Logger::LOG_IMPLEMENTATION ) \
      {                                                                 \
         std::stringstream ss__impl_macro;                                          \
         ss__impl_macro << msg;                                                     \
         _logPrivate.writeln( ss__impl_macro, nll::core::Logger::LOG_IMPLEMENTATION );      \
      }                                                                 \
   }
   #endif

   #ifndef NLL_LOG_INFO
   #define NLL_LOG_INFO(msg)                                         \
   {                                                                 \
      auto& _logPrivate = nll::core::LoggerNll::instance();                  \
      if ( _logPrivate.getLogLevel() <= nll::core::Logger::LOG_INFO )        \
      {                                                              \
         std::stringstream ss__impl_macro;                                       \
         ss__impl_macro << msg;                                                  \
         _logPrivate.writeln( ss__impl_macro, nll::core::Logger::LOG_INFO );             \
      }                                                              \
   }
   #endif

   #ifndef NLL_LOG_ERROR
   #define NLL_LOG_ERROR(msg)                                        \
   {                                                                 \
      auto& _logPrivate = nll::core::LoggerNll::instance();                  \
      if ( _logPrivate.getLogLevel() <= nll::core::Logger::LOG_ERROR )       \
      {                                                              \
         std::stringstream ss__impl_macro;                                       \
         ss__impl_macro << msg;                                                  \
         _logPrivate.writeln( ss__impl_macro, nll::core::Logger::LOG_ERROR );            \
      }                                                              \
   }
   #endif

   #ifndef NLL_LOG_WARNING
   #define NLL_LOG_WARNING(msg)                                      \
   {                                                                 \
      auto& _logPrivate = nll::core::LoggerNll::instance();                  \
      if ( _logPrivate.getLogLevel() <= nll::core::Logger::LOG_WARNING )     \
      {                                                              \
         std::stringstream ss__impl_macro;                                       \
         ss__impl_macro << msg;                                                  \
         _logPrivate.writeln( ss__impl_macro, nll::core::Logger::LOG_WARNING );          \
      }                                                              \
   }
   #endif
}
}

#endif
