add_subdirectory(proto)

set(common 
flatc.h
parameters.h
parameters.cpp
)


add_library(ps_client 
  ps_client.cpp ps_client.h ${common} 
)

target_link_libraries(
    ps_client
    Folly::folly
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
)

add_executable(ps_client_test
    ps_client_test.cpp 
)
target_link_libraries(ps_client_test
    ps_client
)


add_executable(ps_server
    ps_server.cpp 
    ps_server.h 
    cache_ps_impl.h
    ${common}
)
target_link_libraries(ps_server
    Folly::folly
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
)
