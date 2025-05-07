add_executable(xmh-array-bench main.cu)
target_link_libraries(xmh-array-bench PUBLIC pthread)
target_include_directories(xmh-array-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include)
target_include_directories(xmh-array-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include/freestanding/include)
target_compile_options(xmh-array-bench PUBLIC -Wno-unused-function -Wno-aligned-new -Wno-format)

target_link_libraries(xmh-array-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/build/lib/libnvm.so)
set_target_properties(xmh-array-bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(xmh-array-bench PROPERTIES CUDA_ARCHITECTURES OFF)


add_executable(xmh-cuda-test cuda-test.cu)
target_link_libraries(xmh-cuda-test PUBLIC ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/build/lib/libnvm.so)
set_target_properties(xmh-cuda-test PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(xmh-cuda-test PROPERTIES CUDA_ARCHITECTURES OFF)
