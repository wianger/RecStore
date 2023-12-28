set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architecture version")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

add_library(recstore_pytorch MODULE 
    recstore.cu 
    merge.cu 
    gpu_cache_nohash.cu 
    shm.cc 
    kg_controller.cc
    IPCTensor.cc
)


target_link_libraries(recstore_pytorch PUBLIC ${TORCH_LIBRARIES} )
target_include_directories(recstore_pytorch PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(recstore_pytorch PUBLIC base gpu_cache TBB::tbb)
set_target_properties(recstore_pytorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch PROPERTIES CUDA_ARCHITECTURES OFF)
target_link_libraries(recstore_pytorch PRIVATE Python3::Python)


add_library(recstore_pytorch_test MODULE 
    torchcpp_test.cc
)
target_link_libraries(recstore_pytorch_test PUBLIC ${TORCH_LIBRARIES} )
target_include_directories(recstore_pytorch_test PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(recstore_pytorch_test  PRIVATE Python3::Python)


# target_compile_options(recstore_pytorch PUBLIC "-fsanitize=thread")
# target_link_options(recstore_pytorch PUBLIC "-fsanitize=thread")