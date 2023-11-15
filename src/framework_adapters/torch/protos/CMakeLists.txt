set(PROTOS
    ${CMAKE_CURRENT_SOURCE_DIR}/IPCGPUMemoryHandle.proto
)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTOS})
# grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${CMAKE_CURRENT_BINARY_DIR} ${PROTOS})

add_library(ipc_gpu_memory_proto ${PROTO_SRCS} ${GRPC_SRCS})
target_include_directories(ipc_gpu_memory_proto  PUBLIC .  PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
