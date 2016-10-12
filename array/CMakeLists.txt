cmake_minimum_required(VERSION 2.8)

set(LIBNAME    "array")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)
set(ApiFile    "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h")
set(Headers    "array.h" "${ApiFile}"
			      "array-traits.h"
			      "index-mapper.h"
			      "forward.h"
			      "static-vector.h"
			      "traits.h"
			      "range.h"
			      "array-processor.h"
			      "apply-vector.h"
				  "array-io.h"
				  "static-vector-op.h"
				  "vector-helpers.h"
				  "array-flatten.h"
				  "static-vector-math.h"
				  "array-op.h"
				  "array-op-common.h"
				  "array-op-dummy.h"
				  "array-op-blas.h"
				  "math.h"
               )
set(Sources    "array.cpp"
               )

set(BlasWrappers    "wrapper-blas-f2c.h"
					"wrapper-blas-f2c.cpp"
                    "wrapper-blas-dispatcher.h"
                    "wrapper-blas-dispatcher.cpp"
					"wrapper-blas.h"
					"wrapper-blas.cpp"
					"wrapper-default-configuration.cpp"
					"wrapper-common.h"
               )

set(CudaFiles  ""
			   )

source_group(Cuda FILES ${CudaFiles})
source_group(BlasWrappers FILES ${BlasWrappers})
source_group(Sources FILES ${Sources} ${Headers})

MESSAGE("OpenBLAS=" ${OpenBLAS_LIB})


add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${Headers} ${BlasWrappers})
target_link_libraries(${LIBNAME} infra ${NLL_EXTERNAL_LIB} ${OpenBLAS_LIB})



include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
	BASE_NAME ${LIBNAME_CAP}
	EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
	EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
	STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)