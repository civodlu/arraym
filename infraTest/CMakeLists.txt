file(GLOB_RECURSE MYINCLUDES "*.h")
file(GLOB_RECURSE MYSOURCES "*.cpp")
source_group("Sources" FILES ${MYINCLUDES} ${MYSOURCES})

set(EXECNAME infraTest)
set(DATA_TEST_PATH "${CMAKE_CURRENT_SOURCE_DIR}/")
add_executable(${EXECNAME} ${MYINCLUDES} ${MYSOURCES})
add_definitions( -DDATA_TEST_PATH="${DATA_TEST_PATH}/")
target_link_libraries(${EXECNAME} infra tester)

set_target_properties( ${EXECNAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")