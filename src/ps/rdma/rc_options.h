#pragma once

#include <folly/portability/GFlags.h>

DECLARE_int32(rdma_rc_qps_per_client_per_shard);
DECLARE_int32(rdma_rc_mtu_bytes);
DECLARE_int32(rdma_rc_target_response_mtu);
DECLARE_int32(rdma_rc_target_request_mtu);
DECLARE_int32(rdma_rc_request_slot_bytes);
DECLARE_int32(rdma_rc_response_slot_bytes);
DECLARE_int32(rdma_wait_timeout_ms);
DECLARE_string(rdma_rc_namespace);
