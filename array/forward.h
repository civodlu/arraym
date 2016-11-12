#pragma once

#pragma warning(disable:4503)   // decorated names too long
#pragma warning(disable:4512)   // default assignment can't be generated
#pragma warning(disable:4520)   // multiple default constructor false positive

#include <type_traits>
#include <iostream>
#include <vector>
#include <memory>
#include <array>
#include <stdint.h>

#ifdef _MSC_VER
# define FORCE_INLINE   __forceinline
#else
# define FORCE_INLINE   __attribute__((always_inline))
#endif

#ifndef DECLARE_NAMESPACE_NLL
#define DECLARE_NAMESPACE_NLL    namespace nll{ namespace core {
#endif

#ifndef DECLARE_NAMESPACE_END
#define DECLARE_NAMESPACE_END    } }
#endif

#ifndef NAMESPACE_NLL
#define NAMESPACE_NLL nll::core
#endif

# define NLL_ASSERT_IMPL( _e, _s ) \
          if ( ( _e ) == 0 ) {                                      \
            std::stringstream ss_impl;                              \
            ss_impl << _s;                                          \
            std::cout << "------------" << std::endl;				     \
	         std::cout << "Error : " << ss_impl.str() << std::endl;  \
	         std::cout << "  Location : " << __FILE__ << std::endl;  \
	         std::cout << "  Line     : " << __LINE__ << std::endl;  \
            throw std::runtime_error( ss_impl.str() ); } 0    // 0 to force the ";"

#ifdef NDEBUG
# define NLL_FAST_ASSERT(xxx, msg) (void*)0
#else
/// to be used only for assert checked in debug builds
# define NLL_FAST_ASSERT(xxx, msg)  NLL_ASSERT_IMPL(xxx, msg)
#endif

// see ref http://stackoverflow.com/questions/7090998/portable-unused-parameter-macro-used-on-function-signature-for-c-and-c

/**
@brief Specify an unused parameter and remove the compiler warning
*/
#ifdef UNUSED
# elif defined(__GNUC__)
#  define UNUSED(x) UNUSED_ ## x __attribute__((unused))
# elif defined(__LCLINT__)
#  define UNUSED(x) /*@unused@*/ x
# elif defined(__cplusplus)
#  define UNUSED(x)
# else
#  define UNUSED(x) x
#endif

DECLARE_NAMESPACE_NLL
using ui32 = uint32_t;
DECLARE_NAMESPACE_END

#include <array/array-api.h>
#include "dot.h"
#include "static-vector.h"
#include "static-vector-math.h"
#include "static-vector-op.h"
#include "traits.h"
#include "index-mapper.h"
#include "memory.h"
#include "array-traits.h"
#include "array.h"
#include "array-processor.h"