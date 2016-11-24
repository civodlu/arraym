cmake_minimum_required(VERSION 2.8)

project(Array CXX C)

set(APPLICATION_PLATFORM WIN64)  # this can be used in infra to link with the correct windows libraries

include(LibHints.cmake)

# disallow in-source build
MACRO(MACRO_ENSURE_OUT_OF_SOURCE_BUILD MSG)
     STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${CMAKE_BINARY_DIR}" insource)
     GET_FILENAME_COMPONENT(PARENTDIR ${CMAKE_SOURCE_DIR} PATH)
     STRING(COMPARE EQUAL "${CMAKE_SOURCE_DIR}"
     "${PARENTDIR}" insourcesubdir)
    IF(insource OR insourcesubdir)
        MESSAGE(FATAL_ERROR "${MSG}")
    ENDIF(insource OR insourcesubdir)
ENDMACRO(MACRO_ENSURE_OUT_OF_SOURCE_BUILD)

macro_ensure_out_of_source_build("${PROJECT_NAME} requires an out of source build. Please create a separate build directory and run 'cmake /path/to/${PROJECT_NAME} [options]' there.")

#---------------------- Boost -------------------------------------
include(DefineOptions.cmake)

# path configuration
set(LIB_TYPE SHARED)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build")
set(LIBRARY_OUTPUT_PATH "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})  # output all binaries in the same folder

include_directories(${CMAKE_CURRENT_SOURCE_DIR} # in project headers
                    ${CMAKE_BINARY_DIR}   # configuration headers
					)
               
if (${WITH_OPENBLAS})
   find_package(OpenBLAS REQUIRED)
   include_directories(${OpenBLAS_INCLUDE_DIR})
endif()

if (${WITH_CLANG})
   find_package(ClangTools REQUIRED)
endif()

# platform configuration
if(UNIX)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fopenmp")
endif()

if (MSVC)
   add_compile_options(/fp:fast /fp:except- /openmp /MP /W4 /D${APPLICATION_PLATFORM})
endif()

add_definitions(-DPATH_TO_TEST_DATA="${PATH_TO_TEST_DATA}" -D_SCL_SECURE_NO_WARNINGS)
add_definitions(-DBOOST_ALL_DYN_LINK) #BOOST_ALL_DYN_LINK use the dynamic lib for boost


add_subdirectory (array)
add_subdirectory (tester)

if (${BUILD_DOCUMENTATION})
	find_package(Doxygen)
	if(NOT DOXYGEN_FOUND)
         message(FATAL_ERROR "Doxygen is needed to build the documentation.")
    endif()
	set(doxyfile ${CMAKE_CURRENT_SOURCE_DIR}/array.doxygen)
		
	add_custom_target(doxygen
                      COMMAND ${DOXYGEN_EXECUTABLE} ${doxyfile}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM)
endif()

if (${CLANG_FORMAT_FOUND})
   file(GLOB_RECURSE  FILES_FORMAT ${CMAKE_CURRENT_SOURCE_DIR}/array/*.h ${CMAKE_CURRENT_SOURCE_DIR}/array/*.cpp)
   message(${FILES_FORMAT})

   add_custom_target(clang_format
                      COMMAND "${CLANG_FORMAT_BIN}" -i -style=file ${FILES_FORMAT}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM)
endif()

if (${CLANG_TIDY_FOUND})
   file(GLOB  FILES_FORMAT ${CMAKE_CURRENT_SOURCE_DIR}/array/*.cpp)
   message(${FILES_FORMAT})

   add_custom_target(clang_tidy
                      COMMAND "${CLANG_TIDY_BIN}" ${FILES_FORMAT} -checks=*,-clang-diagnostic-inconsistent-dllimport,-google-readability-namespace-comments,-llvm-namespace-comment,-google-readability-casting,-readability-inconsistent-declaration-parameter-name -- -std=c++11 -I${CMAKE_CURRENT_SOURCE_DIR} -I${CMAKE_BINARY_DIR} -I${OpenBLAS_INCLUDE_DIR}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                      COMMENT "Generating API documentation with Doxygen"
                      VERBATIM)
endif()