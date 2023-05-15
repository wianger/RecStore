# 
# Copyright (c) 2020, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#      http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#

cmake_minimum_required(VERSION 3.8)

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
