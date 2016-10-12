cmake_minimum_required(VERSION 2.8)

project(Array CXX C)

# global needed variables
set(APPLICATION_NAME ${PROJECT_NAME})
set(APPLICATION_VERSION_MAJOR "0")
set(APPLICATION_VERSION_MINOR "0")
set(APPLICATION_VERSION_PATCH "1")
set(APPLICATION_VERSION "${APPLICATION_VERSION_MAJOR}.${APPLICATION_VERSION_MINOR}.${APPLICATION_VERSION_PATCH}")
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
find_package(Boost REQUIRED filesystem system)
message(Boost_INCLUDE_DIRS= ${Boost_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

include(DefineOptions.cmake)

# path configuration
set(LIB_TYPE SHARED)
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}/build")
set(LIBRARY_OUTPUT_PATH "${EXECUTABLE_OUTPUT_PATH}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})  # output all binaries in the same folder

include_directories(${PROJECT_SOURCE_DIR} # in project headers
                    ${CMAKE_BINARY_DIR}   # configuration headers
					${CMAKE_CURRENT_BINARY_DIR}/zlib) # zconf.h
               
if (${WITH_OPENBLAS})
   find_package(OpenBLAS REQUIRED)
   include_directories(${OpenBLAS_INCLUDE_DIR})
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

add_subdirectory (zlib)
add_subdirectory (infra)
add_subdirectory (tester)
add_subdirectory (infraTest)


set(CORE_STR "Core")
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(TARGET infra  PROPERTY FOLDER ${CORE_STR})
set_property(TARGET infraTest  PROPERTY FOLDER ${CORE_STR})
set_property(TARGET tester  PROPERTY FOLDER ${CORE_STR})