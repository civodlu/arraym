file(GLOB_RECURSE MYINCLUDES "*.h")
file(GLOB_RECURSE MYSOURCES "*.cpp")
file(GLOB_RECURSE MYSOURCES_CUDA "*.cu")
file(GLOB_RECURSE MYSOURCES_CUDAH "*.cuh")
source_group("Sources" FILES ${MYINCLUDES} ${MYSOURCES})

set(EXECNAME arrayTest)
set(DATA_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")

if(${CUDA_FOUND})
	message ("WITH CUDA!=" ${MYSOURCES_CUDA} ${MYSOURCES_CUDAH})
	cuda_add_executable(${EXECNAME} ${MYINCLUDES} ${MYSOURCES} ${MYSOURCES_CUDA} ${MYSOURCES_CUDAH})
else()
	add_executable(${EXECNAME} ${MYINCLUDES} ${MYSOURCES})
endif()

add_definitions( -DDATA_TEST_PATH="${DATA_TEST_PATH}/")
target_link_libraries(${EXECNAME} array tester)
set_target_properties( ${EXECNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")