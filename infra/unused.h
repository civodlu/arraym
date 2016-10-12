#ifndef NLL_INFRA_UNUSED_H_
# define NLL_INFRA_UNUSED_H_

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

#endif