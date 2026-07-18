# Headers and lib must match: third_party liburing 2.13 uses sq.ring_entries
# in inline get_sqe. System liburing.so.2 often resolves to an older .so.2.2
# whose queue_init leaves ring_entries=0 → get_sqe always nullptr.
set(RECSTORE_LIBURING_A
    ${PROJECT_SOURCE_DIR}/third_party/liburing/src/liburing.a)
if(NOT EXISTS "${RECSTORE_LIBURING_A}")
  message(FATAL_ERROR
    "Missing ${RECSTORE_LIBURING_A}; build third_party/liburing first")
endif()

add_library(iouring_backend io_uring_backend.cpp)

target_include_directories(iouring_backend
  PUBLIC
    ${PROJECT_SOURCE_DIR}/third_party/liburing/src/include
)

target_link_libraries(iouring_backend
  PUBLIC
    base
    boost_context
    ${RECSTORE_LIBURING_A}
)
