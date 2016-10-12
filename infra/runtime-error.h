#ifndef NLL_CORE_RUNTIME_ERROR_H_
# define NLL_CORE_RUNTIME_ERROR_H_

namespace nll
{
namespace core
{
   /**
    @ingroup core
    @brief Display the call stack to a stream
    */
   INFRA_API void printCallStack( std::ostream& o );

   /**
    @brief Runtime exception
    @note the stack trace is recorded
    */
   class INFRA_API RuntimeError : public std::exception
   {
   public:
      RuntimeError( const std::string& msg, bool captureStackTrace = true, bool displayStackTrace = true ) throw();

      virtual const char* what() const throw() override;
      virtual const char* trace() const throw();

       ~RuntimeError() throw()
       {}

   private:
      std::string _msg;
      std::string _callstack;
   };
}
}

#endif
