cmake_minimum_required(VERSION 2.8)

set(LIBNAME    "infra")
string(TOUPPER ${LIBNAME} LIBNAME_CAP)
set(ApiFile    "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h")
set(ConfigFile "${CMAKE_BINARY_DIR}/${LIBNAME}/config.h")
set(Headers    "types.h" "${ApiFile}" "${ConfigFile}"
               "memory-info.h"
               "forward.h"
               "utility-pure.h"
               "runtime-error.h"
               "ensure.h"
               "io.h"
               "symbol.h"
               "timer.h"
               "indent.h"
			   "assert.h"
			   "context.h"
			   "unused.h"
			   "zfstream.h"
			   "logger.h"
               )
set(Sources    "memory-info.cpp"
               "runtime-error.cpp"
               "symbol.cpp"
               "indent.cpp"
			   "zfstream.cpp"
               )
set(CudaFiles  "test-cuda.h" "test-cuda.cu")

include (CheckIncludeFiles)
if (${WITH_OMP})
   check_include_files(omp.h WITH_OMP)
endif()
check_include_files(execinfo.h NLL_HAS_EXECINFO)
check_include_files(zlib.h WITH_ZLIB)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/config.h.in ${ConfigFile})

if (${WITH_CUDA})
   find_package(CUDA)
   if (CUDA_FOUND)
      include_directories(${CUDA_INCLUDE_DIRS})
	  message("CUDA libs=" ${CUDA_LIBRARIES})
	  message("CUDA includes=" ${CUDA_INCLUDE_DIRS})
      cuda_add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${Headers} ${CudaFiles} ${CUDA_LIBRARIES})
   else()
      message( FATAL_ERROR "CUDA installation could not be located" )
   endif()
else()
   message( "CUDA disabled" )
   add_library(${LIBNAME} ${LIB_TYPE} ${Sources} ${Headers})
endif()

if (${WITH_ZLIB})
	target_link_libraries(${LIBNAME} zlib)
endif()

include (GenerateExportHeader)
generate_export_header( ${LIBNAME}
	BASE_NAME ${LIBNAME_CAP}
	EXPORT_MACRO_NAME ${LIBNAME_CAP}_API
	EXPORT_FILE_NAME "${CMAKE_BINARY_DIR}/${LIBNAME}/${LIBNAME}-api.h"
	STATIC_DEFINE ${LIBNAME_CAP}_BUILT_AS_STATIC
)