#include "ps/rdma/rc_transport.h"

#include <cstring>
#include <stdexcept>

#include <folly/portability/GFlags.h>

#include "ps/rdma/rc_options.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);

namespace petps {
namespace {

constexpr std::uint64_t kSubmitDescriptorWrId = 1;
constexpr std::uint64_t kSubmitCommitWrId     = 2;
constexpr std::uint64_t kResponsePayloadWrId  = 3;
constexpr std::uint64_t kResponseStatusWrId   = 4;

std::size_t ServerLaneBytes(const RcTransportConfig& config) {
  return static_cast<std::size_t>(config.num_clients) *
         config.request_slot_bytes;
}

std::size_t ClientLaneBytes(const RcTransportConfig& config) {
  return config.response_slot_bytes + config.request_slot_bytes;
}

std::size_t ClientShardLaneOffset(const RcTransportConfig& config) {
  return static_cast<std::size_t>(config.shard_id) * ClientLaneBytes(config);
}

std::size_t
ServerRequestOffset(const RcTransportConfig& config, int client_id) {
  return static_cast<std::size_t>(client_id) * config.request_slot_bytes;
}

std::size_t ClientResponseOffset(const RcTransportConfig& config) {
  return ClientShardLaneOffset(config);
}

std::size_t ClientRequestStagingOffset(const RcTransportConfig& config) {
  return ClientShardLaneOffset(config) + config.response_slot_bytes;
}

std::uint64_t RequestCommitOffset(const RcTransportConfig& config) {
  return config.request_slot_bytes - Align64(sizeof(CommitWord));
}

std::uint64_t ResponseStatusOffset(const RcTransportConfig& config) {
  return config.response_slot_bytes - Align64(sizeof(StatusWord));
}

RawVerbsConfig MakeRawConfig(
    const RcTransportConfig& config,
    int local_lane,
    std::size_t local_region_bytes,
    bool is_client,
    int only_node_id) {
  RawVerbsConfig raw;
  raw.global_id          = FLAGS_global_id;
  raw.local_lane         = local_lane;
  raw.remote_lane        = local_lane;
  raw.only_node_id       = only_node_id;
  raw.num_servers        = FLAGS_num_server_processes;
  raw.num_clients        = FLAGS_num_client_processes;
  raw.connect_to_servers = is_client;
  raw.connect_to_clients = !is_client;
  raw.local_region_bytes = local_region_bytes;
  (void)config;
  return raw;
}

void PollWrite(RawVerbsTransport* verbs, std::uint64_t wr_id) {
  RawVerbsCompletion completion;
  if (!verbs->Poll(&completion, FLAGS_rdma_wait_timeout_ms)) {
    throw std::runtime_error("RC verbs write completion timeout");
  }
  if (completion.wr_id != wr_id) {
    throw std::runtime_error("unexpected RC verbs write completion");
  }
}

void ValidateClientId(const RcTransportConfig& config, int client_id) {
  if (client_id < 0 || client_id >= config.num_clients) {
    throw std::runtime_error("client_id out of range");
  }
}

} // namespace

RcShardClientTransport::RcShardClientTransport(const RcTransportConfig& config)
    : config_(config), server_node_id_(config.shard_id) {
  ValidateClientId(config_, config_.client_id);
  if (server_node_id_ < 0 || server_node_id_ >= FLAGS_num_server_processes) {
    throw std::runtime_error("server shard id out of global node range");
  }
  lanes_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  for (int qp = 0; qp < config_.qps_per_client_per_shard; ++qp) {
    Lane lane;
    const std::size_t local_bytes =
        static_cast<std::size_t>(FLAGS_num_server_processes) *
        ClientLaneBytes(config_);
    RawVerbsConfig raw =
        MakeRawConfig(config_, qp, local_bytes, true, server_node_id_);
    raw.reserved_region_offset = ClientShardLaneOffset(config_);
    raw.reserved_region_bytes  = ClientLaneBytes(config_);
    lane.verbs                 = std::make_unique<RawVerbsTransport>(raw);
    lane.response_slot         = lane.verbs->LocalPointer(GlobalAddress{
        static_cast<std::uint16_t>(FLAGS_global_id),
        static_cast<std::uint64_t>(ClientResponseOffset(config_)),
    });
    lane.request_staging       = lane.verbs->LocalPointer(GlobalAddress{
        static_cast<std::uint16_t>(FLAGS_global_id),
        static_cast<std::uint64_t>(ClientRequestStagingOffset(config_)),
    });
    std::memset(lane.response_slot, 0, config_.response_slot_bytes);
    std::memset(lane.request_staging, 0, config_.request_slot_bytes);
    lane.verbs->PublishAndConnect();
    lanes_.push_back(std::move(lane));
  }
}

RcShardClientTransport::~RcShardClientTransport() = default;

RcShardClientTransport::Lane& RcShardClientTransport::LaneAt(int qp_index) {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

const RcShardClientTransport::Lane&
RcShardClientTransport::LaneAt(int qp_index) const {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

RcClientQpView RcShardClientTransport::OpenQp(int qp_index) {
  const Lane& lane   = LaneAt(qp_index);
  auto* request_slot = static_cast<char*>(lane.request_staging);
  auto* descriptor   = reinterpret_cast<RequestDescriptor*>(request_slot);
  auto* payload      = request_slot + Align64(sizeof(RequestDescriptor));
  auto* commit       = reinterpret_cast<CommitWord*>(
      request_slot + config_.request_slot_bytes - Align64(sizeof(CommitWord)));

  auto* response_slot    = static_cast<char*>(lane.response_slot);
  auto* response_payload = response_slot;
  auto* status           = reinterpret_cast<StatusWord*>(
      response_payload + config_.response_slot_bytes -
      Align64(sizeof(StatusWord)));
  const int slot_index =
      config_.client_id * config_.qps_per_client_per_shard + qp_index;

  return RcClientQpView{
      qp_index,
      slot_index,
      request_slot,
      descriptor,
      payload,
      commit,
      response_slot,
      response_payload,
      status,
  };
}

void RcShardClientTransport::SubmitRequest(
    const RcClientQpView& view,
    const RequestDescriptor& descriptor,
    const void* payload,
    std::size_t payload_bytes) {
  Lane& lane             = LaneAt(view.qp_index);
  auto* request_slot     = static_cast<char*>(lane.request_staging);
  auto* local_descriptor = reinterpret_cast<RequestDescriptor*>(request_slot);
  auto* local_payload    = request_slot + Align64(sizeof(RequestDescriptor));
  auto* local_commit     = reinterpret_cast<CommitWord*>(
      request_slot + RequestCommitOffset(config_));
  *local_descriptor = descriptor;
  if (payload_bytes > 0) {
    std::memcpy(local_payload, payload, payload_bytes);
  }
  local_commit->seq.store(descriptor.seq, std::memory_order_release);
  local_commit->state.store(kRcSlotReady, std::memory_order_release);

  const std::uint64_t remote_request_offset =
      ServerRequestOffset(config_, config_.client_id);
  lane.verbs->Write(
      request_slot,
      GlobalAddress{
          static_cast<std::uint16_t>(server_node_id_),
          remote_request_offset,
      },
      Align64(sizeof(RequestDescriptor)) + payload_bytes,
      kSubmitDescriptorWrId,
      true);
  PollWrite(lane.verbs.get(), kSubmitDescriptorWrId);

  lane.verbs->Write(
      local_commit,
      GlobalAddress{
          static_cast<std::uint16_t>(server_node_id_),
          remote_request_offset + RequestCommitOffset(config_),
      },
      sizeof(CommitWord),
      kSubmitCommitWrId,
      true);
  PollWrite(lane.verbs.get(), kSubmitCommitWrId);
}

void RcShardClientTransport::ClearRequestSlot(const RcClientQpView& view) {
  auto* commit = view.commit;
  commit->state.store(0, std::memory_order_release);
}

RcShardServerTransport::RcShardServerTransport(const RcTransportConfig& config)
    : config_(config) {
  if (FLAGS_global_id < 0 || FLAGS_global_id >= FLAGS_num_server_processes) {
    throw std::runtime_error("server global_id out of range");
  }
  lanes_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  for (int qp = 0; qp < config_.qps_per_client_per_shard; ++qp) {
    Lane lane;
    const std::size_t local_bytes =
        ServerLaneBytes(config_) +
        static_cast<std::size_t>(config_.num_clients) *
            config_.response_slot_bytes;
    lane.verbs = std::make_unique<RawVerbsTransport>(
        MakeRawConfig(config_, qp, local_bytes, false, -1));
    lane.request_slots =
        lane.verbs->AllocateRegistered(ServerLaneBytes(config_));
    std::memset(lane.request_slots, 0, ServerLaneBytes(config_));
    lane.response_staging.reserve(
        static_cast<std::size_t>(config_.num_clients));
    for (int client = 0; client < config_.num_clients; ++client) {
      void* slot = lane.verbs->AllocateRegistered(config_.response_slot_bytes);
      std::memset(slot, 0, config_.response_slot_bytes);
      lane.response_staging.push_back(slot);
    }
    lane.verbs->PublishAndConnect();
    lanes_.push_back(std::move(lane));
  }
}

RcShardServerTransport::~RcShardServerTransport() = default;

RcShardServerTransport::Lane& RcShardServerTransport::LaneAt(int qp_index) {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

const RcShardServerTransport::Lane&
RcShardServerTransport::LaneAt(int qp_index) const {
  if (qp_index < 0 || qp_index >= config_.qps_per_client_per_shard) {
    throw std::runtime_error("qp_index out of range");
  }
  return lanes_.at(static_cast<std::size_t>(qp_index));
}

int RcShardServerTransport::TotalSlots() const {
  return config_.num_clients * config_.qps_per_client_per_shard;
}

void* RcShardServerTransport::RequestSlot(int slot_index) const {
  if (slot_index < 0 || slot_index >= TotalSlots()) {
    throw std::runtime_error("slot_index out of range");
  }
  const int client_id = slot_index / config_.qps_per_client_per_shard;
  const int qp_index  = slot_index % config_.qps_per_client_per_shard;
  const Lane& lane    = LaneAt(qp_index);
  return static_cast<char*>(lane.request_slots) +
         ServerRequestOffset(config_, client_id);
}

RequestDescriptor*
RcShardServerTransport::RequestDescriptorAt(int slot_index) const {
  return reinterpret_cast<RequestDescriptor*>(RequestSlot(slot_index));
}

char* RcShardServerTransport::RequestPayloadAt(int slot_index) const {
  return static_cast<char*>(RequestSlot(slot_index)) +
         Align64(sizeof(RequestDescriptor));
}

CommitWord* RcShardServerTransport::RequestCommitAt(int slot_index) const {
  return reinterpret_cast<CommitWord*>(
      static_cast<char*>(RequestSlot(slot_index)) +
      RequestCommitOffset(config_));
}

RcShardServerTransport::ResponseView
RcShardServerTransport::OpenClientResponse(int client_id, int qp_index) {
  ValidateClientId(config_, client_id);
  Lane& lane = LaneAt(qp_index);
  auto* slot = static_cast<char*>(
      lane.response_staging.at(static_cast<std::size_t>(client_id)));
  auto* payload = static_cast<char*>(slot);
  auto* status =
      reinterpret_cast<StatusWord*>(payload + ResponseStatusOffset(config_));
  return ResponseView{slot, payload, status};
}

void RcShardServerTransport::CompleteResponse(
    int client_id,
    int qp_index,
    const ResponseView& response,
    std::uint64_t seq) {
  ValidateClientId(config_, client_id);
  Lane& lane = LaneAt(qp_index);
  response.status->seq.store(seq, std::memory_order_release);
  response.status->state.store(kRcSlotDone, std::memory_order_release);

  const int client_node_id = FLAGS_num_server_processes + client_id;
  if (response.status->response_bytes > 0) {
    lane.verbs->Write(
        response.payload,
        GlobalAddress{
            static_cast<std::uint16_t>(client_node_id),
            static_cast<std::uint64_t>(ClientResponseOffset(config_)),
        },
        response.status->response_bytes,
        kResponsePayloadWrId,
        true);
    PollWrite(lane.verbs.get(), kResponsePayloadWrId);
  }

  lane.verbs->Write(
      response.status,
      GlobalAddress{
          static_cast<std::uint16_t>(client_node_id),
          static_cast<std::uint64_t>(
              ClientResponseOffset(config_) + ResponseStatusOffset(config_)),
      },
      sizeof(StatusWord),
      kResponseStatusWrId,
      true);
  PollWrite(lane.verbs.get(), kResponseStatusWrId);
}

} // namespace petps
