cmake_minimum_required(VERSION 2.8)

set(LIBNAME    "array")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)
set(ApiFile    "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h")
set(Headers    array.h 
               ${ApiFile}
               array.h
               array-processor.h
               array-traits.h
               array-exp.h
			   array-noexp.h
               array-op-naive.h
			   array-op-blas.h
			   matrix-op-blas.h
			   matrix-op-naive.h
               op-naive.h
               forward.h
               index-mapper.h
               memory-contiguous.h
               memory-slice.h
               static-vector.h
               static-vector-math.h
               static-vector-op.h
               traits.h   
               )
               
set(BLAS       wrapper-cblas.cpp
               wrapper-cblas.h
               wrapper-common.h
               blas-dispatcher.h
               blas-dispatcher.cpp
               blas-wrapper.h
               blas-wrapper.cpp
			   blas-wrapper-default.cpp)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${CMAKE_BINARY_DIR}/${LIBNAME}/config.h)
source_group(Sources FILES ${Sources} ${Headers})
source_group(BLAS FILES ${BLAS})

MESSAGE("OpenBLAS=" ${OpenBLAS_LIB})


add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${BLAS} ${Headers} ${BlasWrappers})
target_link_libraries(${LIBNAME} ${NLL_EXTERNAL_LIB} ${OpenBLAS_LIB})



include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
	BASE_NAME ${LIBNAME_CAP}
	EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
	EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
	STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)

add_subdirectory (tests)