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

#ifndef NLL_CORE_INFRA_FORWARD_H_
# define NLL_CORE_INFRA_FORWARD_H_

#ifdef _MSC_VER
# define __func__ __FUNCTION__                  // standard way to obtain the current function name (rather than the VS specific one)
# define RESTRICT       __declspec(restrict)    // Applied to a function declaration or definition that returns a pointer type and tells the compiler that the function returns an object that will not be aliased with any other pointers
# define NLL_ALIGN_16   __declspec(align(16))   // 16 bytes aligned memory
# define FORCE_INLINE   __forceinline           // force the inlining
#else
# define RESTRICT       __restrict__
# define NLL_ALIGN_16   __attribute__((aligned(16)))
# define FORCE_INLINE   __attribute__((always_inline))
#endif

//
// define this constant to disable all multithreading in quick loops. This is a workaround for the problematic implementation
// of OpenMP in VS2010 with spin lock.
// See http://social.msdn.microsoft.com/Forums/en-AU/parallelcppnative/thread/528479c8-fb70-4b05-83ce-7a552fd49895
//
#define NLL_NOT_MULTITHREADED_FOR_QUICK_OPERATIONS

//
// define this in debug mode to find memory leaks
//
//#define NLL_FIND_MEMORY_LEAK

#ifdef NLL_FIND_MEMORY_LEAK
# ifdef _MSC_VER
#  ifdef _DEBUG
#   pragma message( "NLL_FIND_MEMORY_LEAK activated" )
#   define NLL_FIND_MEMORY_LEAK_ACTIVATED
#   define _CRTDBG_MAP_ALLOC
#   include <stdlib.h>
#   include <crtdbg.h>
#   define DEBUG_NEW     new(_NORMAL_BLOCK, __FILE__, __LINE__)
#   define new           DEBUG_NEW
#   define malloc(s)     _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#   define PLACEMENT_NEW   new
#  else
#  pragma message( "NLL_FIND_MEMORY_LEAK has not effect in release mode!" )
#  endif
# else
#  pragma message( "NLL_FIND_MEMORY_LEAK only supported in windows!" )
#endif
#endif

// standard includes
# include <assert.h>
# include <time.h>
# include <fstream>
# include <stdexcept>
# include <cstdlib>
# include <memory>
# include <map>
# include <list>
# include <stack>
# include <vector>
# include <queue>
# include <numeric>
# include <string>
# include <limits>
# include <iostream>
# include <sstream>
# include <typeinfo>
# include <utility>
# include <cstring>
# include <set>
# include <type_traits>
# include <cmath>
# include <algorithm>
# include <iomanip>
# include <ostream>
# include <chrono>

// lib's headers
#include <type_traits>
#include <infra/config.h>
#include <infra/infra-api.h>
#include "types.h"
#include "unused.h"
#include "memory-info.h"
#include "test-cuda.h"
#include "runtime-error.h"
#include "assert.h"
#include "ensure.h"
#include "utility-pure.h"
#include "io.h"
#include "symbol.h"
#include "singleton.h"
#include "timer.h"
#include "indent.h"
#include "logger.h"
#include "context.h"
#include "zfstream.h"

#endif
