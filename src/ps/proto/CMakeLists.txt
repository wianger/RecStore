set(PROTOS
    ${CMAKE_CURRENT_SOURCE_DIR}/ps.proto
)

set(PROTO_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR})
file(MAKE_DIRECTORY ${PROTO_SRC_DIR})
include_directories(${PROTO_SRC_DIR})

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_SRC_DIR} ${PROTOS})
grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_SRC_DIR} ${PROTOS})

message(STATUS "ps_proto_src is ${PROTO_SRCS}")
message(STATUS "ps_grpc_src is ${GRPC_SRCS}")


add_library(ps_proto ${PROTO_SRCS} ${GRPC_SRCS})

target_include_directories(ps_proto PUBLIC .  PUBLIC ${PROTO_SRC_DIR})
