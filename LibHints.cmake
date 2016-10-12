set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules/")

#QT5
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "C:/Qt/Qt5.5.0/5.5/msvc2013_64")

#Boost
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${PROJECT_SOURCE_DIR}/external/Boost-1.59")

# Python
set(CMAKE_PREFIX_PATH "C:/WinPython-64bit-2.7.9.5/python-2.7.9.amd64" ${CMAKE_PREFIX_PATH} )

# DCMTK
#set(CMAKE_PREFIX_PATH "C:/Program Files/DCMTK" ${CMAKE_PREFIX_PATH} )
#set(DCMTK_DIR "${CMAKE_SOURCE_DIR}/external/DCMTK_release_x64/")

set(DCMTK_DIR "${CMAKE_SOURCE_DIR}/external/dcmtk-3.6.0-bin/")

set(NUMPY_DIR "${CMAKE_SOURCE_DIR}/external/boost.numpy/")
