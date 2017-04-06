SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/openblas-base
  /usr/local/include
  /usr/local/include/openblas-base
  /opt/OpenBLAS/include
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  ${OpenBLAS_DIR}/include
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/openblas-base
        /lib64/
        /usr/lib
        /usr/lib/openblas-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /opt/OpenBLAS/lib
        $ENV{OpenBLAS}cd
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
        ${OpenBLAS_DIR}/lib
 )
 
 SET(Open_BLAS_BIN_SEARCH_PATHS
		/opt/OpenBLAS/bin
		${OpenBLAS_DIR}/bin/
 )

set(CMAKE_FIND_LIBRARY_SUFFIXES_ORIG ${CMAKE_FIND_LIBRARY_SUFFIXES})
set(CMAKE_FIND_LIBRARY_PREFIXES_ORIG ${CMAKE_FIND_LIBRARY_PREFIXES})
 
set(CMAKE_FIND_LIBRARY_PREFIXES lib)

MESSAGE("OpenBLAS_DIR=" ${OpenBLAS_DIR})
FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

set(CMAKE_FIND_LIBRARY_SUFFIXES .dll .so)
FIND_LIBRARY(OpenBLAS_BIN_DIR NAMES openblas PATHS ${Open_BLAS_BIN_SEARCH_PATHS})

set(CMAKE_FIND_LIBRARY_SUFFIXES ${CMAKE_FIND_LIBRARY_SUFFIXES_ORIG} "")
set(CMAKE_FIND_LIBRARY_PREFIXES ${CMAKE_FIND_LIBRARY_PREFIXES_ORIG} "")

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(WARNING "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(WARNING "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check Bin files
IF(NOT OpenBLAS_BIN_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(WARNING "Could not find OpenBLAS runtime binaries. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
    MESSAGE(STATUS "Found OpenBLAS binaries: ${OpenBLAS_BIN_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
	OpenBLAS_BIN_DIR
    OpenBLAS
)