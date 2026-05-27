add_library(ps_base parameters.cpp cache_ps_impl.h parameters.h)

add_library(ps_common Postoffice.cc config.cc)
target_link_libraries(ps_common PUBLIC base)
