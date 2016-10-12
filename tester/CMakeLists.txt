file(GLOB_RECURSE MYINCLUDES "*.h")
file(GLOB_RECURSE MYSOURCES "*.cpp")
source_group("Sources" FILES ${MYINCLUDES} ${MYSOURCES})

set(LIBNAME    "tester")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)

add_library(${LIBNAME} ${LIB_TYPE} ${MYINCLUDES} ${MYSOURCES})

include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
	BASE_NAME ${LIBNAME_CAP}
	EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
	EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
	STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)

target_link_libraries(${LIBNAME} infra)