# cmake_minimum_required(VERSION 3.10)
project(recstore_tf_ops)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_include())"
    OUTPUT_VARIABLE TENSORFLOW_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TENSORFLOW_INCLUDE_RESULT
)

execute_process(
    COMMAND ${Python_EXECUTABLE} -c "import tensorflow as tf; print(tf.sysconfig.get_lib())"
    OUTPUT_VARIABLE TENSORFLOW_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TENSORFLOW_LIB_RESULT
)

if(NOT TENSORFLOW_INCLUDE_RESULT EQUAL 0 OR NOT TENSORFLOW_LIB_RESULT EQUAL 0)
    message(FATAL_ERROR
        "RECSTORE_ENABLE_TENSORFLOW requires the selected Python interpreter "
        "to have the tensorflow package installed.")
endif()

file(GLOB TENSORFLOW_FRAMEWORK_SO ${TENSORFLOW_LIB_DIR}/libtensorflow_framework.so*)
file(GLOB TENSORFLOW_CC_SO ${TENSORFLOW_LIB_DIR}/libtensorflow_cc.so*)

set(TENSORFLOW_LIBRARIES
    ${TENSORFLOW_FRAMEWORK_SO}
    ${TENSORFLOW_CC_SO}
)

# --- Build shared library ---
add_library(recstore_tf_ops SHARED
    op_tensorflow.cc
    ../op.cc
    ../common/local_shm_kv_client_op.cc
    ../common/hierkv_local_runtime.cc
    ../common/op_runtime_support.cc
    ../common/local_shm_op_component.cc
    ../common/ps_client_config_adapter.cc
    ../../ps/client_factory.cc
    ${PROJECT_SOURCE_DIR}/src/ps/rdma/rdma_ps_client_adapter.cc
)

if(USE_FAKE_KVCLIENT)
    target_compile_definitions(recstore_tf_ops PRIVATE USE_FAKE_KVCLIENT)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src/framework)

# --- Include directories ---
target_include_directories(recstore_tf_ops PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${TENSORFLOW_INCLUDE_DIR}
    ${Python_INCLUDE_DIRS}
)

# --- Link libraries ---
target_link_libraries(recstore_tf_ops PRIVATE
    ${TENSORFLOW_LIBRARIES}
    wq_ps_client
    grpc_ps_client
    dist_grpc_ps_client
    brpc_ps_client
    dist_brpc_ps_client
    local_shm_ps_client
    protobuf::libprotobuf
    gRPC::grpc++
    glog::glog
)

# --- Set C++ standard ---
set_target_properties(recstore_tf_ops PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME "_recstore_tf_ops"
)

# --- Tests ---
add_test(
    NAME tf_client_test
    COMMAND ${Python_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/src/test/framework/tensorflow/client_test.py
            "$<TARGET_FILE:recstore_tf_ops>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/tensorflow
)

# add_test(
#     NAME dist_emb_unittest
#     COMMAND ${Python3_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
# )

# set_tests_properties(dist_emb_unittest PROPERTIES
#     WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/tensorflow
# )
