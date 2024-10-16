project(UVM_LIBTORCH LANGUAGES CXX CUDA C)
set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "CUDA architecture version")


list(APPEND CMAKE_PREFIX_PATH "/home/xieminhui/.local/lib/python3.10/site-packages/torch/share/cmake/Torch/")

find_package(Torch REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development)


add_executable(uvm_libtorch uvm_libtorch.cu )
target_include_directories(uvm_libtorch PUBLIC ${TORCH_INCLUDE_DIRS})
target_link_libraries(uvm_libtorch PUBLIC ${TORCH_LIBRARIES} )
target_link_libraries(uvm_libtorch PRIVATE Python3::Python)
# target_link_libraries(uvm_libtorch PUBLIC profiler)

set_target_properties(uvm_libtorch PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(uvm_libtorch PROPERTIES CUDA_ARCHITECTURES OFF)
