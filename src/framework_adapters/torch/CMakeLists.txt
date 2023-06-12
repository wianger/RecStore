set(Torch_DIR "/opt/conda/lib/python3.8/site-packages/torch/share/cmake/Torch/")
find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)

add_library(recstore_pytorch MODULE cache.cu)
target_link_libraries(recstore_pytorch PUBLIC ${TORCH_LIBRARIES} gpu_cache)
target_include_directories(recstore_pytorch PUBLIC ${TORCH_INCLUDE_DIRS})
set_target_properties(recstore_pytorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(recstore_pytorch PROPERTIES CUDA_ARCHITECTURES OFF)

target_link_libraries(recstore_pytorch  PRIVATE Python3::Python)