add_library(ps_common Postoffice.cc config.cc)
target_link_libraries(ps_common PUBLIC base memcached)
target_include_directories(ps_common PUBLIC ${PROJECT_BINARY_DIR}/third_party/Mayfly-main)