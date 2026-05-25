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
DEFINE_string(rdma_rc_namespace,
              "",
              "Override RC write shared-memory namespace");
