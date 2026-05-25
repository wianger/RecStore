option(ENABLE_RDMA_INTEGRATION_TESTS "Enable RDMA integration tests" OFF)

if(ENABLE_RDMA_INTEGRATION_TESTS)
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
endif()

add_library(wq_ps_client
    petps_client.cc
    allshards_ps_client.cc
    rc_options.cc
    rc_transport.cc
    raw_verbs_transport.cc
    ${PROJECT_SOURCE_DIR}/src/ps/base/parameters.cpp
)
                         
target_link_libraries(wq_ps_client PUBLIC base ps_common mayfly Folly::folly pthread rt)
set(warp_wq_ps_client -Wl,--whole-archive wq_ps_client -Wl,--no-whole-archive CACHE INTERNAL "wq_ps_client")

add_executable(petps_server
    petps_server.cc
    rc_options.cc
    rc_transport.cc
    raw_verbs_transport.cc
)
target_link_libraries(petps_server PUBLIC
    base
    ps_base
    kv_engine
    sparse_tensor
    ps_common
    mayfly
    TBB::tbb
    Folly::folly
    pthread
    rt
)

add_executable(
    petps_integration_test
    ${PROJECT_SOURCE_DIR}/src/test/ps/rdma/petps_integration_test.cpp
)
add_dependencies(petps_integration_test petps_server)
target_link_libraries(petps_integration_test PUBLIC
    wq_ps_client
    base
    gtest
    Folly::folly
    pthread
)

if(ENABLE_RDMA_INTEGRATION_TESTS)
add_test(
        NAME petps_single_shard_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.PutGetRoundTripSingleShard:PetPSIntegrationTest.MissingKeysReturnZeroSlots
    )
    set_tests_properties(
        petps_single_shard_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_multi_shard_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_integration.py
                --server-count 2
                --client-count 1
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.PutGetRoundTripMultiShard
    )
    set_tests_properties(
        petps_multi_shard_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 60
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_multiclient_stress_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_multiclient_stress.py
                --server-count 1
                --client-count 2
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressSingleShard:PetPSIntegrationTest.AsyncGetPrefetchStressSingleShard
    )
    set_tests_properties(
        petps_multiclient_stress_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 90
            LABELS "rdma_integration"
    )

    add_test(
        NAME petps_multishard_multiclient_stress_test
        COMMAND ${Python3_EXECUTABLE}
                ${PROJECT_SOURCE_DIR}/src/test/scripts/run_petps_multiclient_stress.py
                --server-count 2
                --client-count 2
                --config-path ${PROJECT_SOURCE_DIR}/src/test/configs/recstore_config.rdma_multishard_test.json
                --test-binary $<TARGET_FILE:petps_integration_test>
                --gtest-filter=PetPSIntegrationTest.RepeatedPutGetStressMultiShard
    )
    set_tests_properties(
        petps_multishard_multiclient_stress_test
        PROPERTIES
            SKIP_RETURN_CODE 77
            TIMEOUT 120
            LABELS "rdma_integration"
    )
endif()
