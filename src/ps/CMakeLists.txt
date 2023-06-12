add_subdirectory(proto)

set(common 
    flatc.h
    parameters.h
    parameters.cpp
)


add_library(grpc_ps_client 
    grpc_ps_client.cpp grpc_ps_client.h ${common} 
)

target_link_libraries(
    grpc_ps_client PUBLIC
    base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
)

add_executable(grpc_ps_client_test
    grpc_ps_client_test.cpp 
)
target_link_libraries(grpc_ps_client_test PUBLIC
    grpc_ps_client
)


add_executable(grpc_ps_server
    grpc_ps_server.cpp 
    grpc_ps_server.h 
    cache_ps_impl.h
    ${common}
)
target_link_libraries(grpc_ps_server PUBLIC
    base
    ps_proto
    gRPC::grpc++_reflection
    gRPC::grpc++
    protobuf::libprotobuf
    stdc++fs
)
