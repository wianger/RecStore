#include "ps/rdma/rc_options.h"

DEFINE_int32(rdma_rc_qps_per_client_per_shard,
             32,
             "RC write QPs per client per shard");
DEFINE_int32(rdma_rc_mtu_bytes, 4096, "RC write logical MTU bytes");
DEFINE_int32(rdma_rc_target_response_mtu,
             200,
             "RC write target GET response MTU count");
DEFINE_int32(rdma_rc_target_request_mtu,
             200,
             "RC write target PUT request MTU count");
DEFINE_int32(rdma_rc_request_slot_bytes,
             1 << 20,
             "RC write request slot bytes");
DEFINE_int32(rdma_rc_response_slot_bytes,
             1 << 20,
             "RC write response slot bytes");
DEFINE_int32(rdma_wait_timeout_ms,
             60000,
             "RC write single RPC wait timeout in milliseconds");
DEFINE_int32(rdma_rc_profile_interval_ms,
             0,
             "RC write profiling summary interval in milliseconds; 0 disables "
             "profiling");
DEFINE_int32(rdma_rc_server_coroutines_per_thread,
             1,
             "RC write server coroutine count per poll thread. Values greater "
             "than 1 enable cooperative slot scanning inside each poll thread");
DEFINE_string(rdma_rc_namespace,
              "",
              "Override RC write shared-memory namespace");
DEFINE_string(rdma_rc_fake_get_mode,
              "none",
              "Benchmark-only fake GET mode: none, status_only, or "
              "payload_memset");
DEFINE_bool(rdma_rc_skip_client_copy,
            false,
            "Benchmark-only option to skip copying GET response payload from "
            "the RC response slot to the user receive buffer");
