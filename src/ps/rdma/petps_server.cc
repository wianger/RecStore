#include <folly/init/Init.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/bind_core.h"
#include "base/config.h"
#include "base/log.h"
#include "base/timer.h"
#include "memory/shm_file.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/rdma/rc_options.h"
#include "ps/rdma/rc_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

DEFINE_string(config_path, "", "config file path");
DEFINE_int32(thread_num, 1, "RC write poll thread count");
DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DEFINE_int32(value_size, 128, "embedding row bytes");
DEFINE_int32(max_kv_num_per_request, 500, "max keys per request");
DEFINE_bool(use_dram, false, "unused compatibility flag");
DEFINE_int32(numa_id, 0, "NUMA node id for mmap and core binding");

namespace {

bool ShouldTraceRdmaGet() {
  static const bool enabled = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::uint64_t RdmaGetTraceInterval() {
  static const std::uint64_t interval = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE_INTERVAL");
    if (env == nullptr) {
      return std::uint64_t{5000};
    }
    const auto parsed =
        static_cast<std::uint64_t>(std::strtoull(env, nullptr, 10));
    return parsed == 0 ? std::uint64_t{5000} : parsed;
  }();
  return interval;
}

std::string TimestampNow() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::to_string(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::string NamespaceToken() {
  if (!FLAGS_rdma_rc_namespace.empty()) {
    return FLAGS_rdma_rc_namespace;
  }
  if (const char* env = std::getenv("RECSTORE_MEMCACHED_NAMESPACE")) {
    return env;
  }
  return "default";
}

int ResolveShardId(const nlohmann::json& config) {
  const int default_shard = FLAGS_global_id;
  if (!config.contains("cache_ps") || !config["cache_ps"].is_object()) {
    return default_shard;
  }
  const auto& cache_ps = config["cache_ps"];
  if (cache_ps.contains("servers") && cache_ps["servers"].is_array()) {
    for (const auto& server : cache_ps["servers"]) {
      if (server.value("shard", -1) == FLAGS_global_id) {
        return server.value("shard", default_shard);
      }
    }
  }
  return default_shard;
}

void NormalizeDramValuePath(nlohmann::json* base_kv_config) {
  if (base_kv_config == nullptr || !base_kv_config->is_object()) {
    return;
  }
  if (!base_kv_config->contains("value") ||
      !(*base_kv_config)["value"].is_object()) {
    return;
  }
  auto& value_cfg = (*base_kv_config)["value"];
  const std::string value_type =
      value_cfg.value("type", std::string("DRAM_VALUE_STORE"));
  if (value_type != "DRAM_VALUE_STORE") {
    return;
  }
  const std::string path = value_cfg.value("path", std::string());
  if (path.empty() || path.rfind("/dev/shm", 0) == 0) {
    return;
  }
  value_cfg["path"] = "/dev/shm/recstore_rdma_rc_" + TimestampNow() + "/value";
}

class PetPSServer {
public:
  PetPSServer(CachePS* cache_ps,
              int thread_count,
              int shard_id,
              const std::string& namespace_token)
      : cache_ps_(cache_ps), thread_count_(thread_count), shard_id_(shard_id) {
    petps::RcTransportConfig config;
    config.shard_id                 = shard_id_;
    config.num_clients              = FLAGS_num_client_processes;
    config.qps_per_client_per_shard = FLAGS_rdma_rc_qps_per_client_per_shard;
    config.request_slot_bytes =
        static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes);
    config.response_slot_bytes =
        static_cast<std::size_t>(FLAGS_rdma_rc_response_slot_bytes);
    config.namespace_token = namespace_token;
    transport_ = std::make_unique<petps::RcShardServerTransport>(config);
    last_seq_.assign(
        static_cast<std::size_t>(transport_->TotalSlots()), std::uint64_t{0});
  }

  void Run() {
    for (int i = 0; i < thread_count_; ++i) {
      threads_.emplace_back(&PetPSServer::PollingThread, this, i);
    }
  }

private:
  void HandleGet(const petps::RequestDescriptor& descriptor,
                 const char* payload,
                 petps::RcShardServerTransport::ResponseView* response,
                 int thread_id) {
    base::ConstArray<std::uint64_t> keys(
        reinterpret_cast<const std::uint64_t*>(payload), descriptor.key_count);
    const bool ok = cache_ps_->GetParameterFlat(
        keys,
        reinterpret_cast<float*>(response->payload),
        descriptor.key_count,
        descriptor.embedding_dim,
        thread_id);
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kValueSizeMismatch);
    response->status->response_bytes =
        static_cast<std::uint32_t>(descriptor.response_bytes);
  }

  void HandlePut(const petps::RequestDescriptor& descriptor,
                 const char* payload,
                 petps::RcShardServerTransport::ResponseView* response,
                 int thread_id) {
    const auto* reader =
        reinterpret_cast<const ParameterCompressReader*>(payload);
    if (!reader->Valid(static_cast<int>(descriptor.payload_bytes))) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }
    for (int i = 0; i < reader->item_size(); ++i) {
      cache_ps_->PutSingleParameter(reader->item(i), thread_id);
    }
    response->status->status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    response->status->response_bytes = 0;
  }

  void HandleUpdate(const petps::RequestDescriptor& descriptor,
                    const char* payload,
                    petps::RcShardServerTransport::ResponseView* response,
                    int thread_id) {
    const std::string_view table_name = petps::DescriptorTableName(descriptor);
    if (table_name.empty()) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    const auto* reader =
        reinterpret_cast<const ParameterCompressReader*>(payload);
    if (!reader->Valid(static_cast<int>(descriptor.payload_bytes))) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    const bool ok = cache_ps_->UpdateParameter(
        std::string(table_name), reader, static_cast<unsigned>(thread_id));
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kInvalidPayload);
    response->status->response_bytes = 0;
  }

  void HandleInitTable(const petps::RequestDescriptor& descriptor,
                       const char* payload,
                       petps::RcShardServerTransport::ResponseView* response) {
    const std::string_view table_name = petps::DescriptorTableName(descriptor);
    if (table_name.empty() ||
        descriptor.payload_bytes != petps::InitTablePayloadBytes()) {
      response->status->status =
          static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
      response->status->response_bytes = 0;
      return;
    }

    std::uint64_t num_embeddings = 0;
    std::uint64_t embedding_dim  = 0;
    std::memcpy(&num_embeddings, payload, sizeof(num_embeddings));
    std::memcpy(&embedding_dim,
                payload + sizeof(num_embeddings),
                sizeof(embedding_dim));
    const bool ok = cache_ps_->InitTable(
        std::string(table_name), num_embeddings, embedding_dim);
    response->status->status = static_cast<std::int32_t>(
        ok ? petps::RpcStatus::kOk : petps::RpcStatus::kInvalidPayload);
    response->status->response_bytes = 0;
  }

  void PollingThread(int thread_id) {
    base::auto_bind_core();
    LOG(INFO) << "component=rdma_server event=polling_thread_ready thread_id="
              << thread_id;
    const int total_slots = transport_->TotalSlots();
    while (true) {
      for (int slot = thread_id; slot < total_slots; slot += thread_count_) {
        auto* commit = transport_->RequestCommitAt(slot);
        if (commit->state.load(std::memory_order_acquire) !=
            petps::kRcSlotReady) {
          continue;
        }
        const std::uint64_t seq = commit->seq.load(std::memory_order_acquire);
        if (seq == 0 || seq == last_seq_[static_cast<std::size_t>(slot)]) {
          continue;
        }

        auto* descriptor = transport_->RequestDescriptorAt(slot);
        std::string error;
        if (!petps::ValidateRequestDescriptor(
                *descriptor,
                transport_->config().request_slot_bytes,
                transport_->config().response_slot_bytes,
                &error)) {
          LOG(ERROR)
              << "component=rdma_rc_server event=invalid_descriptor"
              << " shard=" << shard_id_ << " slot=" << slot
              << " thread_id=" << thread_id << " seq=" << seq
              << " descriptor_seq=" << descriptor->seq
              << " client_id=" << descriptor->client_id
              << " qp=" << descriptor->qp_index << " op=" << descriptor->op
              << " key_count=" << descriptor->key_count
              << " payload_bytes=" << descriptor->payload_bytes
              << " response_bytes=" << descriptor->response_bytes << " error=\""
              << error << "\"";
          last_seq_[static_cast<std::size_t>(slot)] = seq;
          commit->state.store(0, std::memory_order_release);
          continue;
        }

        auto response = transport_->OpenClientResponse(
            descriptor->client_id, descriptor->qp_index);
        const char* payload = transport_->RequestPayloadAt(slot);
        VLOG(1) << "component=rdma_rc_server event=consume shard=" << shard_id_
                << " slot=" << slot << " client_id=" << descriptor->client_id
                << " qp=" << descriptor->qp_index << " seq=" << seq << " op="
                << descriptor->op << " key_count=" << descriptor->key_count
                << " payload_bytes=" << descriptor->payload_bytes
                << " response_bytes=" << descriptor->response_bytes;
        response.status->status =
            static_cast<std::int32_t>(petps::RpcStatus::kInvalidPayload);
        response.status->response_bytes = 0;

        if (descriptor->shard_id != static_cast<std::uint32_t>(shard_id_)) {
          LOG(ERROR) << "component=rdma_rc_server event=wrong_shard"
                     << " expected_shard=" << shard_id_ << " actual_shard="
                     << descriptor->shard_id << " slot=" << slot
                     << " client_id=" << descriptor->client_id
                     << " qp=" << descriptor->qp_index << " seq=" << seq
                     << " op=" << descriptor->op
                     << " key_count=" << descriptor->key_count;
          response.status->status =
              static_cast<std::int32_t>(petps::RpcStatus::kWrongShard);
        } else if (descriptor->op ==
                   static_cast<std::uint16_t>(petps::RcOp::kGet)) {
          HandleGet(*descriptor, payload, &response, thread_id);
        } else if (descriptor->op ==
                   static_cast<std::uint16_t>(petps::RcOp::kPut)) {
          HandlePut(*descriptor, payload, &response, thread_id);
        } else if (descriptor->op ==
                   static_cast<std::uint16_t>(petps::RcOp::kUpdate)) {
          HandleUpdate(*descriptor, payload, &response, thread_id);
        } else if (descriptor->op ==
                   static_cast<std::uint16_t>(petps::RcOp::kInitTable)) {
          HandleInitTable(*descriptor, payload, &response);
        }

        std::atomic_thread_fence(std::memory_order_release);
        transport_->CompleteResponse(
            descriptor->client_id, descriptor->qp_index, response, seq);
        VLOG(1) << "component=rdma_rc_server event=complete shard=" << shard_id_
                << " slot=" << slot << " client_id=" << descriptor->client_id
                << " qp=" << descriptor->qp_index << " seq=" << seq
                << " status=" << response.status->status
                << " response_bytes=" << response.status->response_bytes;
        last_seq_[static_cast<std::size_t>(slot)] = seq;
      }
      std::this_thread::yield();
    }
  }

  CachePS* cache_ps_ = nullptr;
  int thread_count_  = 1;
  int shard_id_      = 0;
  std::unique_ptr<petps::RcShardServerTransport> transport_;
  std::vector<std::thread> threads_;
  std::vector<std::uint64_t> last_seq_;
};

} // namespace

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv);
  if (ShouldTraceRdmaGet()) {
    std::cerr << "component=rdma_get_trace side=server event=enabled interval="
              << RdmaGetTraceInterval() << std::endl;
  }
  xmh::Reporter::StartReportThread();

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(FLAGS_use_dram);
  base::PMMmapRegisterCenter::GetConfig().numa_id  = FLAGS_numa_id;

  extern int global_socket_id;
  global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  const std::string config_path =
      FLAGS_config_path.empty()
          ? base::ResolveRecStoreConfigPath().string()
          : FLAGS_config_path;
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    LOG(FATAL) << "Cannot open config file: " << config_path;
  }

  nlohmann::json config;
  config_file >> config;
  if (config.contains("cache_ps") && config["cache_ps"].is_object() &&
      config["cache_ps"].contains("base_kv_config")) {
    NormalizeDramValuePath(&config["cache_ps"]["base_kv_config"]);
  }
  if (config.contains("distributed_client") &&
      config["distributed_client"].is_object() &&
      config["distributed_client"].contains("base_kv_config")) {
    NormalizeDramValuePath(&config["distributed_client"]["base_kv_config"]);
  }
  auto cache_ps      = std::make_unique<CachePS>(config["cache_ps"]);
  const int shard_id = ResolveShardId(config);
  auto ps            = std::make_unique<PetPSServer>(
      cache_ps.get(), FLAGS_thread_num, shard_id, NamespaceToken());
  ps->Run();
  while (true) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
