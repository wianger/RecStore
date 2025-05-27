set(RECSTORE_PATH ${PROJECT_SOURCE_DIR})
configure_file(${PROJECT_SOURCE_DIR}/src/recstore_config.h.in
	${PROJECT_BINARY_DIR}/recstore_config.h
)
include_directories(${PROJECT_BINARY_DIR})


add_subdirectory(base)

add_subdirectory(ps_common)
add_subdirectory(grpc_ps)
# add_subdirectory(rdma_ps)

add_subdirectory(memory)
add_subdirectory(storage)

add_subdirectory(kg)
add_subdirectory(test)

add_subdirectory(small_tests)


add_subdirectory(framework)

add_subdirectory(executable/uvm)