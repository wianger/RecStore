
add_executable(xmh-integrity-util integrity.c transfer.c util.c)
target_include_directories(xmh-integrity-util PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/include)
target_link_libraries(xmh-integrity-util PUBLIC ${PROJECT_SOURCE_DIR}/third_party/ssd-gpu-dma/build/lib/libnvm.so)
target_link_libraries(xmh-integrity-util PUBLIC pthread)