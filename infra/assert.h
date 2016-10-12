#ifndef NLL_CORE_ASSERT_H_
# define NLL_CORE_ASSERT_H_

# define NLL_ASSERT_IMPL( _e, _s ) \
          if ( ( _e ) == 0 ) {                                      \
            std::stringstream ss_impl;                              \
            ss_impl << _s;                                          \
            std::cout << "------------" << std::endl;				     \
	         std::cout << "Error : " << ss_impl.str() << std::endl;  \
	         std::cout << "  Location : " << __FILE__ << std::endl;  \
	         std::cout << "  Line     : " << __LINE__ << std::endl;  \
            throw nll::core::RuntimeError( ss_impl.str() ); } 0    // 0 to force the ";"

#ifdef NDEBUG
# define NLL_FAST_ASSERT(xxx, msg) (void*)0
#else
/// to be used only for assert checked in debug builds
# define NLL_FAST_ASSERT(xxx, msg)  NLL_ASSERT_IMPL(xxx, msg)
#endif

#endif