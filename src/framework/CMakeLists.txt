option(USE_FAKE_KVCLIENT "Use FakeKVClientOp for testing" OFF)

find_package(Python 3.10 COMPONENTS Interpreter Development REQUIRED)

if(ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/gpu)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/pytorch)
# add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/tensorflow)

if(USE_FAKE_KVCLIENT)
    message(STATUS "Using FakeKVClientOp for testing!")
    add_definitions(-DUSE_FAKE_KVCLIENT)
endif()

add_test(
    NAME pytorch_client_test
    COMMAND ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
)

add_test(
    NAME pytorch_client_test_rdma_basic
    COMMAND ${CMAKE_COMMAND} -E env
            RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
            RECSTORE_CLIENT_TEST_PHASE=basic
            RECSTORE_USE_LOCAL_MEMCACHED=auto
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
)

add_test(
    NAME pytorch_client_test_rdma
    COMMAND ${CMAKE_COMMAND} -E env
            RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
            RECSTORE_CLIENT_TEST_PHASE=full
            RECSTORE_USE_LOCAL_MEMCACHED=auto
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
)

add_test(
    NAME pytorch_client_test_rdma_auto
    COMMAND ${CMAKE_COMMAND} -E env
            RECSTORE_CONFIG=${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.op_rdma.json
            RECSTORE_CLIENT_TEST_PHASE=full
            RECSTORE_USE_LOCAL_MEMCACHED=auto
            ${Python_EXECUTABLE} ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch/test_client.py
            "$<TARGET_FILE:recstore_torch_ops>"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/test/framework/pytorch
)

set_tests_properties(
    pytorch_client_test_rdma_basic
    pytorch_client_test_rdma
    pytorch_client_test_rdma_auto
    PROPERTIES
        SKIP_RETURN_CODE 77
        TIMEOUT 90
        LABELS "rdma_integration"
)

add_test(
    NAME dist_emb_unittest
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_dist_emb
)

set_tests_properties(dist_emb_unittest PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

add_test(
    NAME ebc_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper
)

set_tests_properties(ebc_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)

add_test(
    NAME ebc_multiprocess_precision_test
    COMMAND ${Python_EXECUTABLE} -m unittest recstore.unittest.test_ebc_precision_wrapper.TestEBCPrecision.test_multiprocess_precision
)

set_tests_properties(ebc_multiprocess_precision_test PROPERTIES
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/src/python/pytorch
)
