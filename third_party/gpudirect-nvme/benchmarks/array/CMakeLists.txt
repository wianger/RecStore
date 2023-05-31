cmake_minimum_required (VERSION 3.1)
project (libnvm-benchmarks)

set (THREADS_PREFER_PTHREAD_FLAG ON)
find_package (Threads REQUIRED)

find_package (CUDA 8.0 REQUIRED)



make_benchmark (array-benchmark-module array-bench "main.cu;")

make_benchmark_choice (array-benchmark array-benchmark-smartio array-benchmark-module)



# set(CUDA_SEPARABLE_COMPILATION ON)
# cuda_add_executable(xmh-array-bench main.cu)
# target_link_libraries(xmh-array-bench libnvm)
# target_link_libraries(xmh-array-bench pthread)
# target_link_libraries(xmh-array-bench ${CUDA_LIBRARIES})


