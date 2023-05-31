
file (GLOB module_files "src/*.cu" "src/*.cc")
# file (GLOB smartio_files "src/*.cu" "src/*.cc" "smartio/*.cu" "smartio/*.cc")

add_executable(nvm-latency-bench ${module_files})
target_include_directories(nvm-latency-bench PUBLIC include)
target_include_directories(nvm-latency-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/include)
target_link_libraries(nvm-latency-bench PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/build/lib/libnvm.so)
target_link_libraries(nvm-latency-bench PUBLIC pthread)
set_target_properties(nvm-latency-bench PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(nvm-latency-bench PROPERTIES CUDA_ARCHITECTURES OFF)