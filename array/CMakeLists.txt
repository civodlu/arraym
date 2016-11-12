cmake_minimum_required(VERSION 2.8)

set(LIBNAME    "array")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)
set(ApiFile    "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h")
set(Headers    "array.h" "${ApiFile}"
			      "static-vector.h"
				   "forward.h"
               "array-details.h"
               "index-mapper.h"
			   "dot.h"
               )
set(Sources    "dummy.cpp"
               )


source_group(Sources FILES ${Sources} ${Headers})

MESSAGE("OpenBLAS=" ${OpenBLAS_LIB})


add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${Headers} ${BlasWrappers})
target_link_libraries(${LIBNAME} ${NLL_EXTERNAL_LIB} ${OpenBLAS_LIB})



include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
	BASE_NAME ${LIBNAME_CAP}
	EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
	EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
	STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)

add_subdirectory (tests)