#pragma once

#include <folly/portability/GFlags.h>

DECLARE_int32(rdma_rc_qps_per_client_per_shard);
DECLARE_int32(rdma_rc_slots_per_qp);
DECLARE_int32(rdma_rc_mtu_bytes);
DECLARE_int32(rdma_rc_target_response_mtu);
DECLARE_int32(rdma_rc_target_request_mtu);
DECLARE_int32(rdma_rc_request_slot_bytes);
DECLARE_int32(rdma_rc_response_slot_bytes);
DECLARE_int32(rdma_wait_timeout_ms);
DECLARE_int32(rdma_rc_profile_interval_ms);
DECLARE_int32(rdma_rc_server_coroutines_per_thread);
DECLARE_int32(rdma_rc_inline_bytes);
DECLARE_int32(rdma_rc_client_numa_id);
DECLARE_int32(rdma_rc_server_numa_id);
DECLARE_string(rdma_rc_namespace);
DECLARE_string(rdma_rc_fake_get_mode);
DECLARE_bool(rdma_rc_skip_client_copy);
