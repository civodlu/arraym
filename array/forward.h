#pragma once

#include <array/config.h>

/**
 @mainpage Arraym

 The purpose of this library is to propose a generic multidimensional array with
 written in modern C++. The library features:
 - Efficient memory access patterns using processors,
 - Support for BLAS implementations (cuBLAS-XT & OpenBLAS),
 - Uniform type representation for N-d structures, vectors & matrices, including small object
   optimization,
 - Support for non-contiguous memory based arrays,
 - API ease of use, 
 - Interoperability with other array objects: can be wrapped with no data copy in most cases.
 - Basic linear algebra (e.g., inverse, SVD, LU, determinant, least square)
 - cross platform (VS2013, GCC, Clang)

 @section Examples
 The array can simply be accessed using subscripting or indices:
 @code
 Array<float, 2> array(2, 3);
 array(0, 0) = 1;
 array({1, 1}) = 2;
 @endcode

 The array can be filled using the initializer list. In this case the array will be filled
 by dimension (e.g., x, y, z...).

 @code
 Array<float, 2> array(2, 3);
 array = {1, 2, 3, 4, 5, 6};
 assert(array(1, 0) == 2); // array is not filled by memory order but by dimension
 @endcode

 The array has a value based semantics, meaning that we prefer copying the value of the array
 rather than sharing the memory between arrays.

 @code
 Array<float, 2> array1(2, 3);
 Array<float, 2> array2(1, 1)
 array2 = array1; // the array2 current memory is released and the content of array1 is copied
 @endcode

 Although for performance and convenience it is useful to reference a portion of the array and modify
 it as required (@ref ArrayRef). This can be done using a min and max (inclusive) index:
 @code
 Array<float, 2> array(10, 10);
 auto sub_array = array({2, 2}, {5, 5});  // this points to min=(2, 2) to max=(5, 5) of array
 sub_array = 42;  // the referenced array will be updated
 @endcode

 Similarly a range can be used to access sub-array:
 @code
 Array<float, 2> array(10, 10);
 auto sub_array = array(R(2, 3), R(4, 5)); // select area for x = 2..3 and y = 4..5
 sub_array = 42;  // the referenced array will be updated
 @endcode

 while a @ref ArrayRef is in use, the original @ref Array must be kept alive.

 Internally, the arithmetic operations used on arrays are controlled by 3 template: @ref array_use_naive,
 @ref array_use_blas, @ref array_use_vectorization. For a given array, only one of these template must
 have a true value. Implemented with a layered approach:
 - basic linear algebra building blocks for arrays such as @ref details::array_add, have a separate implementation
 for each type (naive, BLAS, vectorized)
 - typical operators are defined. If BLAS is enabled and template expression enabled, template based expression operators
 are selected, if not, simple operators.

 Convenience functions such as @ref mean, @ref min, @ref max, @ref sin, @ref cos, @ref sum, @ref sqrt, @ref sqr,
 @ref log, @ref exp can be applied on each element of the array.
 */

/**
 @page label_1 High level api

 @section label_2 Accessing elements
 Accessing elements can be done elements by elements. For example:
 @code
 Array<float, 2> array(2, 3);
 
 // access element with individual coordinates
 array(0, 0) = 42;

 // access element with coordinates packed in a static vector
 array(vector2ui{1, 1}) = 42;
 @endcode

 For convenience, arrays can be initialized following axis order (not memory order):
 @code
 Array<float, 2> array(3, 2);
 array = {1, 2, 3,
          4, 5, 6};
 @endcode
 
 However these methods are inefficient since they incur many overheads
 (e.g., index calculation) and should be avoided in performance critical code.
 More importantly, we need to be very careful in accessing the array elements
 in a cache friendly manner which may not be done by hand easily. This is why most array operations are
 done in blocks so that we can minimize indexing overhead and take advantage of SIMD operations. Array
 access is abstracted using Processor concept which decides what is the most cache-friendly access
 pattern.

 @section label_3 The processor
 The processor is used to abstract how the array is accessed in order to minimize cache misses and enable
 SIMD and GPU based operations. Array are commonly stored following row-major and column-major (e.g., image below).
 Other access orders can be implemented.

 @image html processor_column_major.png

 @section label_4 Value based semantic
 We chose value based array semantic (as opposed to reference based semantics) as it is less prone to inattention errors.
 This means that by default, we prefer copying arrays and avoid referencing arrays unless explicitly specified. However,
 it is possible to reference arrays when performance is needed.

 @code
 Array<float, 2> array1(3, 2);
 Array<float, 2> array2(3, 2);
 
 // the array1 is fully copied
 array2 = array1;
 @endcode

 The following code create a reference on the main array:
 @code
 Array<float, 2> array(3, 2);

 // array_ref is a reference, if array
 // is destroyed, the reference becomes invalid
 auto array_ref = array({0, 0}, {1, 1});  

 // at this point array_copy will have a copy of the referenced sub-array
 Array<float, 2> array_copy = array_ref;
 @endcode

 @section label_5 Common operations
 Many common operations are defined:
 - arithmetic (e.g., addition, multiplication, sum, mean)
 - boolean (e.g., and, or, any)
 - math (e.g., pow, sqrt, cos)

 @section label_6 Dense linear algebra
 @section label_7 Small array optimization
 @section label_8 Axis-based operations
 @section label_9 GPU based arrays
 @subsection label_9a cuda_ptr<T>
 @subsection label_9b Memory transfers
 @section label_10 BLAS dispatcher
 */

/*
TODO nice to have functions:
 - array<vector3f> * cte [...]
 - rand
 - reshape
 - dot
 - split
*/

#include <array/array-api.h>
#include "wrapper-common.h"

#include "traits.h"
#include "op-naive.h"
#include "static-vector.h"
#include "static-vector-math.h"
#include "static-vector-op.h"
#include "cuda-utils.h"
#include "cuda-kernel.cuh"
#include "allocator-cuda.h"
#include "index-mapper.h"
#include "allocator-static.h"
#include "memory-contiguous.h"
#include "memory-slice.h"
#include "range.h"
#include "enumerate.h"
#include "array-traits.h"
#include "array.h"
#include "array-io.h"
#include "array-chunking.h"
#include "array-processor.h"
#include "array-fill.h"
#include "cuda-array-op.h"
#include "array-op-impl-naive.h"
#include "blas-wrapper.h"
#include "cuda-blas-wrapper.h"
#include "array-op-impl-blas.h"
#include "array-op.h"
#include "array-op-logical.h"
#include "array-op-axis.h"

#include "array-exp.h"
#include "array-noexp.h"

#include "matrix-op-impl-naive.h"
#include "array-transpose.h"
#include "matrix-op-identity.h"
#include "matrix-vector.h"
#include "repmat.h"
#include "concat.h"
#include "linspace.h"
#include "lookup.h"
#include "where.h"
#include "array-dim-iterator.h"
#include "vector-helpers.h"
#include "matrix-op-diag.h"

// BLAS based operations
#include "matrix-op-impl-blas.h"
#include "matrix-op-blas-least-square.h"
#include "matrix-op-blas-inv.h"
#include "matrix-op-blas-svd.h"
#include "matrix-op-blas-lu.h"
#include "matrix-op-blas-cov.h"
#include "matrix-op-blas-det.h"
#include "matrix-op-trace.h"
