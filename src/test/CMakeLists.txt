find_package(folly REQUIRED)

add_executable(test_f14_slab
    test_f14_slab.cpp 
)

target_link_libraries(test_f14_slab PUBLIC
    ${FOLLY_LIBRARIES}
    mimalloc-static
)

target_compile_options(test_f14_slab PUBLIC -msse -msse2 -msse3 -msse4)

add_executable(test_hash_map
    test_hash_map.cpp 
)

target_link_libraries(test_hash_map PUBLIC
    ${FOLLY_LIBRARIES}
)

target_compile_options(test_hash_map PUBLIC -msse -msse2 -msse3 -msse4)
