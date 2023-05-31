set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)


add_executable(nvm-cuda-bench main.cu buffer.cu ctrl.cc settings.cu queue.cu)
target_include_directories(nvm-cuda-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/include)
target_link_libraries(nvm-cuda-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/build/lib/libnvm.so)
set_target_properties(nvm-cuda-bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(nvm-cuda-bench PROPERTIES CUDA_ARCHITECTURES OFF)



add_executable(test_GPUSpdk test_BaMWrapper.cu buffer.cu ctrl.cc settings.cu queue.cu)
target_include_directories(test_GPUSpdk PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/include)
target_link_libraries(test_GPUSpdk PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/build/lib/libnvm.so)
target_link_libraries(test_GPUSpdk PUBLIC Folly::folly)
set_target_properties(test_GPUSpdk PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_GPUSpdk PROPERTIES CUDA_ARCHITECTURES OFF)



add_subdirectory(integrity)
add_subdirectory(latency-benchmark)
add_subdirectory(array)