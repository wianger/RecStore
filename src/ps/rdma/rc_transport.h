#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "ps/rdma/raw_verbs_transport.h"
#include "ps/rdma/rdma_protocol.h"

namespace petps {

struct RcTransportConfig {
  int shard_id    = 0;  // Logical shard served by this transport.
  int client_id   = -1; // Logical client id for response slot selection.
  int num_clients = 1;  // Total client count expected by the server.
  int qps_per_client_per_shard    = 32; // Number of lanes per client per shard.
  std::size_t request_slot_bytes  = 1 << 20; // Bytes per server request slot.
  std::size_t response_slot_bytes = 1 << 20; // Bytes per client response slot.
  std::string namespace_token =
      "default"; // Shared-memory namespace for this run.
};

struct RcClientQpView {
  int qp_index                  = 0; // Lane index local to the client.
  int slot_index                = 0; // Global request slot index on the server.
  void* request_slot            = nullptr; // Base address of the request slot.
  RequestDescriptor* descriptor = nullptr;
  char* payload          = nullptr; // Request payload region after descriptor.
  CommitWord* commit     = nullptr; // Commit word at end of request slot.
  void* response_slot    = nullptr; // Base address of the client response slot.
  char* response_payload = nullptr; // Response payload region before status.
  StatusWord* status     = nullptr; // Status word at end of response slot.
};

class RcShardClientTransport {
public:
  explicit RcShardClientTransport(const RcTransportConfig& config);
  ~RcShardClientTransport();

  RcShardClientTransport(const RcShardClientTransport&)            = delete;
  RcShardClientTransport& operator=(const RcShardClientTransport&) = delete;

  RcClientQpView OpenQp(int qp_index);
  void SubmitRequest(const RcClientQpView& view,
                     const RequestDescriptor& descriptor,
                     const void* payload,
                     std::size_t payload_bytes);
  void ClearRequestSlot(const RcClientQpView& view);
  std::size_t request_slot_bytes() const { return config_.request_slot_bytes; }
  std::size_t response_slot_bytes() const {
    return config_.response_slot_bytes;
  }
  const RcTransportConfig& config() const { return config_; }

private:
  struct Lane {
    std::unique_ptr<RawVerbsTransport>
        verbs; // RC QP and registered memory for this lane.
    void* response_slot   = nullptr; // Local registered response slot.
    void* request_staging = nullptr; // Local registered request staging slot.
  };

  Lane& LaneAt(int qp_index);
  const Lane& LaneAt(int qp_index) const;

  RcTransportConfig config_;
  int server_node_id_ = 0; // Global node id of the target shard server.
  std::vector<Lane> lanes_;
};

class RcShardServerTransport {
public:
  explicit RcShardServerTransport(const RcTransportConfig& config);
  ~RcShardServerTransport();

  RcShardServerTransport(const RcShardServerTransport&)            = delete;
  RcShardServerTransport& operator=(const RcShardServerTransport&) = delete;

  int TotalSlots() const;
  void* RequestSlot(int slot_index) const;
  RequestDescriptor* RequestDescriptorAt(int slot_index) const;
  char* RequestPayloadAt(int slot_index) const;
  CommitWord* RequestCommitAt(int slot_index) const;

  struct ResponseView {
    void* slot         = nullptr; // Base address of the client response slot.
    char* payload      = nullptr; // Response payload region.
    StatusWord* status = nullptr; // Final completion word for this response.
  };

  ResponseView OpenClientResponse(int client_id, int qp_index);
  void CompleteResponse(int client_id,
                        int qp_index,
                        const ResponseView& response,
                        std::uint64_t seq);
  const RcTransportConfig& config() const { return config_; }

private:
  struct Lane {
    std::unique_ptr<RawVerbsTransport>
        verbs;                     // RC QP and registered memory for this lane.
    void* request_slots = nullptr; // Local registered slots for all clients.
    std::vector<void*>
        response_staging; // Per-client registered response staging slots.
  };

  Lane& LaneAt(int qp_index);
  const Lane& LaneAt(int qp_index) const;

  RcTransportConfig config_; // Transport sizing and namespace config.
  std::vector<Lane> lanes_;  // One verbs RC lane per qp_index.
};

} // namespace petps
