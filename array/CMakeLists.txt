cmake_minimum_required(VERSION 2.8)

set(LIBNAME    "array")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)
set(ApiFile    "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h")
set(Headers array.h 
            ${ApiFile}
            array.h
            range.h
            allocator-static.h
            array-processor.h
            array-traits.h
            array-exp.h
            array-noexp.h
			array-fill.h
            array-op-impl-naive.h
            array-op-impl-blas.h
            array-op.h
			array-io.h
            matrix-op-impl-blas.h
            matrix-op-blas-inv.h
            matrix-op-blas-least-square.h
            matrix-op-blas-svd.h
            matrix-op-impl-naive.h
            array-chunking.h
            matrix-op-identity.h
            matrix-vector.h
            repmat.h
            op-naive.h
            forward.h
            index-mapper.h
            memory-contiguous.h
            array-op-axis.h
            matrix-op-blas-lu.h
            matrix-op-blas-cov.h
            memory-slice.h
            static-vector.h
            static-vector-math.h
            static-vector-op.h
            traits.h
			linspace.h
			concat.h
			lookup.h
			where.h
			array-dim-iterator.h
			enumerate.h
			array-op-logical.h
			${CMAKE_BINARY_DIR}/${LIBNAME}/config.h
)
               
set(BLAS    wrapper-cblas.cpp
            wrapper-cblas.h
            wrapper-common.h
            blas-dispatcher.h
            blas-dispatcher.cpp
            blas-wrapper.h
            blas-wrapper.cpp
            wrapper-cublas.h
            wrapper-cublas-xt.cpp
            wrapper-cublas-xt.h
            cublas-common.h
            wrapper-cublas.cpp
            blas-wrapper-default.cpp)

set (CudaKernel cuda-kernel.cu
                cuda-kernel.cuh
				cuda-utils.h
                allocator-cuda.h
				cuda-utils.cpp
				cuda-blas-wrapper.h
				cuda-blas-wrapper.cpp
				cuda-array-op.h
				cuda-array-op.cu)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/${LIBNAME}/config.h)
source_group(Sources FILES ${Sources} ${Headers})
source_group(BLAS FILES ${BLAS})
source_group(CudaKernel FILES ${CudaKernel})

MESSAGE("OpenBLAS=" ${OpenBLAS_LIB})

if(${CUDA_FOUND})
   # CUDA NVBLAS must be before any other BLAS implementation
   set(CUDA_PROPAGATE_HOST_FLAGS ON)
   cuda_add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${BLAS} ${Headers} ${BlasWrappers} ${CudaKernel} OPTIONS --expt-extended-lambda)
else()
   add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${BLAS} ${Headers} ${BlasWrappers})
endif()

target_link_libraries(${LIBNAME} ${NLL_EXTERNAL_LIB} ${CUDA_cublas_LIBRARY} ${CUDA_cudart_static_LIBRARY} ${OpenBLAS_LIB})



include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
   BASE_NAME ${LIBNAME_CAP}
   EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
   EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
   STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)

add_subdirectory (tests)