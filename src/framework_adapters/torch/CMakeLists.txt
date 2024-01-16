set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architecture version")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(GRAD_ASYNC_V1_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    # add_compile_definitions(GRAD_ASYNC_V1_DEBUG)
endif()

set(recstore_pytorch_srcs
    recstore.cu 
    recstore.cc
    merge.cu 
    gpu_cache_nohash.cu 
    shm.cc 
    kg_controller.cc
    IPCTensor.cc
    IPC_barrier.cc
    torch_utils.cc
    parallel_pq.cc
)

add_library(recstore_pytorch MODULE 
    ${recstore_pytorch_srcs}
)

target_include_directories(recstore_pytorch PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(recstore_pytorch PUBLIC ${TORCH_LIBRARIES} )
target_link_libraries(recstore_pytorch PUBLIC base gpu_cache TBB::tbb)
target_link_libraries(recstore_pytorch PRIVATE Python3::Python)
# target_link_libraries(recstore_pytorch PUBLIC profiler)

set_target_properties(recstore_pytorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch PROPERTIES CUDA_ARCHITECTURES OFF)


add_library(torchcpp_test MODULE 
    torchcpp_test.cc
)
target_link_libraries(torchcpp_test PUBLIC ${TORCH_LIBRARIES} )
target_include_directories(torchcpp_test PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(torchcpp_test PRIVATE Python3::Python)


add_executable(recstore_pytorch_test 
    kg_controller_test.cc
    ${recstore_pytorch_srcs}
)
target_include_directories(recstore_pytorch_test PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(recstore_pytorch_test PUBLIC 
${TORCH_LIBRARIES} base Python3::Python
gpu_cache)
# target_link_libraries(recstore_pytorch_test PUBLIC profiler)

set_target_properties(recstore_pytorch_test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch_test PROPERTIES CUDA_ARCHITECTURES OFF)

# target_compile_options(recstore_pytorch_test PUBLIC "-fsanitize=thread")
# target_link_options(recstore_pytorch_test PUBLIC "-fsanitize=thread")