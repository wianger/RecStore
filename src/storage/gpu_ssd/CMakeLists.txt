set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)


add_library(gd_bam BaMWrapper.cu)
target_link_libraries(gd_bam PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/build/lib/libnvm.so)
target_link_libraries(gd_bam PUBLIC Folly::folly)
target_include_directories(gd_bam PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include)
target_include_directories(gd_bam PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include/freestanding/include)
target_compile_options(gd_bam PRIVATE -Wno-unused-function -Wno-aligned-new -Wno-format)
set_target_properties(gd_bam PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(perf_BaM perf_BaM.cu) 
target_link_libraries(perf_BaM PUBLIC gd_bam)
set_target_properties(perf_BaM PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(perf_BaM PROPERTIES CUDA_ARCHITECTURES OFF)


add_subdirectory(array)


add_executable(test_BaM test_BaM.cu)
target_link_libraries(test_BaM gd_bam gtest)
set_target_properties(test_BaM PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_BaM PROPERTIES CUDA_ARCHITECTURES OFF)



add_library(gpu_direct_ssd_raw RawGDWrapper.cu)
target_link_libraries(gpu_direct_ssd_raw PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/build/lib/libnvm.so)
target_link_libraries(gpu_direct_ssd_raw PUBLIC Folly::folly)
target_include_directories(gpu_direct_ssd_raw PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include)
target_include_directories(gpu_direct_ssd_raw PRIVATE ${PROJECT_SOURCE_DIR}/third_party/gpudirect-nvme/include/freestanding/include)
target_compile_options(gpu_direct_ssd_raw PRIVATE -Wno-unused-function -Wno-aligned-new -Wno-format)
set_target_properties(gpu_direct_ssd_raw PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

add_executable(test_RawGD test_RawGD.cu)
target_link_libraries(test_RawGD gpu_direct_ssd_raw gtest)
set_target_properties(test_RawGD PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(test_RawGD PROPERTIES CUDA_ARCHITECTURES OFF)

add_executable(perf_RawGD perf_RawGD.cu)
target_link_libraries(perf_RawGD gpu_direct_ssd_raw gtest)
set_target_properties(perf_RawGD PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(perf_RawGD PROPERTIES CUDA_ARCHITECTURES OFF)

add_executable(conver_file_to_float_array convert_file_to_float_array.cc)
