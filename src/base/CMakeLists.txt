add_library(base base.cc)
target_link_libraries(base PUBLIC Folly::folly ${CITYHASH_LIBRARIES} gflags glog)
target_include_directories(base PUBLIC ${CITYHASH_INCLUDE_DIRS})
target_link_libraries(base PUBLIC protobuf::libprotobuf protobuf::libprotobuf-lite)