# Arraym

The purpose of this library is to propose a generic multidimensional array written in modern C++. The library features:
 - Efficient memory access patterns using processors to abstract memory ordering,
 - Support for BLAS implementations on CPU & GPU (cuBLAS, OpenBLAS),
 - Uniform type representation for N-d structures, vectors & matrices, including small object
   optimization and GPU based types
 - Support for non-contiguous memory based arrays,
 - Simple API, 
 - Interoperability with other array objects: can be wrapped with no data copy in most cases.
 - Basic dense linear algebra (e.g., inverse, SVD, LU, determinant, least square)
 - cross platform (Visual Studio, GCC, Clang)

## Prerequisites

Minimum prerequisites:
- CMake (tested with 3.6.2)
- C++11 compiler (tested with VS2013, g++ 5.4)

The following packages are optional but will disable features if not enabled:
- OpenBLAS : enables dense linear algebra (tested with v0.2.14)
- CUDA V8.0 or greater: enables GPU based multi-dimensional arrays & dense linear algebra

## Installation

Use CMake to locate the dependencies (WITH_CUDA & WITH_OPENBLAS) and generate the makefile for your platform.

## Code Examples

Basic indexing & dense linear algebra:
```cpp
#include <array/forward.h>
using namespace NAMESPACE_NLL;

void api_dense_linear_algebra_subblocks()
{
  Matrix_column_major<float> matrix(6, 7);

  // address only a 2x2 sub-block
  auto sub_2x2 = matrix(R(1, 2), R(2, 3));

  // initialize the sub-block in axis-order fashion
  sub_2x2 = { -1, 4, 5, -8 };

  // compute its inverse
  auto sub_2x2_inv = inv(sub_2x2);
  
  // verify inverse properties ||A * inv(A) - I||_2 == 0
  assert(norm2(sub_2x2 * sub_2x2_inv - identity<float>(2)) < 1e-4f);
}
```

GPU based computations:
```cpp
#include <array/forward.h>
using namespace NAMESPACE_NLL;

void api_cuda_array()
{
  // initialize the memory on the CPU
  Array_column_major<float, 1> cpu_array(4);
  cpu_array = { 1, 2, 3, 4 };

  // transfer to GPU and run calculations
  Array_cuda_column_major<float, 1> gpu_array = cpu_array;
  gpu_array = cos(gpu_array);

  // once all calculations are performed, get the result back on the CPU
  Array_column_major<float, 1> cpu_result = gpu_array;
  for (size_t index : range(cpu_array))
  {
	 assert(fabs(cpu_result(index) - std::cos(cpu_array(index))) < 1e-5f);
  }
}
```

## Tests

Tests are located in array/tests 


