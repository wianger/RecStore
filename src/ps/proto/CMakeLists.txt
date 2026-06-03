set(PROTOS
    ${CMAKE_CURRENT_SOURCE_DIR}/ps.proto
    ${CMAKE_CURRENT_SOURCE_DIR}/rdma_control_plane.proto
)

set(PROTOS_BRPC
    ${CMAKE_CURRENT_SOURCE_DIR}/ps_brpc.proto
)

include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTOS})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${CMAKE_CURRENT_BINARY_DIR} ${PROTOS})

protobuf_generate_cpp(PROTO_BRPC_SRCS PROTO_BRPC_HDRS ${PROTOS_BRPC})

message(STATUS "ps_proto_src is ${PROTO_SRCS}")
message(STATUS "ps_grpc_src is ${GRPC_SRCS}")
message(STATUS "ps_brpc_src is ${PROTO_BRPC_SRCS}")


add_library(ps_proto ${PROTO_SRCS} ${GRPC_SRCS} ${PROTO_BRPC_SRCS})

target_include_directories(ps_proto PUBLIC .  PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(ps_proto PUBLIC gRPC::grpc++ protobuf::libprotobuf)
