#include "ps/rdma/rdma_ps_client_adapter.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>

#include <folly/portability/GFlags.h>
#include <folly/init/Init.h>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/base/config.h"
#include "base/hash.h"
#include "ps/rdma/rdma_common.h"
#include "ps/rdma/rc_options.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);
DECLARE_string(rdma_transport_mode);
DEFINE_string(rdma_transport_mode, "rc_write", "RDMA transport mode: rc_write");

namespace recstore {

namespace {
int ValueSizeHintFromBaseKvConfig(const json& base_kv_config,
                                  int fallback_value_size) {
  if (!base_kv_config.is_object()) {
    return fallback_value_size;
  }
  if (!base_kv_config.contains("value") ||
      !base_kv_config["value"].is_object()) {
    return fallback_value_size;
  }
  return base_kv_config["value"].value(
      "default_value_size_hint", fallback_value_size);
}

std::vector<std::string> ReadProcessArgv() {
  std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
  std::vector<std::string> argv;
  if (!cmdline.is_open()) {
    return argv;
  }

  std::string current;
  char ch = '\0';
  while (cmdline.get(ch)) {
    if (ch == '\0') {
      if (!current.empty()) {
        argv.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(ch);
  }
  if (!current.empty()) {
    argv.push_back(current);
  }
  return argv;
}
} // namespace

int RDMAPSClientAdapter::PartitionKey(uint64_t key) const {
  CHECK_GT(num_shards_, 0);
  if (hash_method_ == "city_hash") {
    return static_cast<int>(GetHash(key) % static_cast<uint64_t>(num_shards_));
  }
  if (hash_method_ == "simple_mod") {
    return static_cast<int>(key % static_cast<uint64_t>(num_shards_));
  }
  throw std::runtime_error("unsupported shard hash method: " + hash_method_);
}

std::vector<RDMAPSClientAdapter::ShardChunk>
RDMAPSClientAdapter::BuildChunks(base::ConstArray<uint64_t> keys) const {
  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::size_t>> shard_positions(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i]);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_positions[static_cast<std::size_t>(shard)].push_back(i);
  }

  std::vector<ShardChunk> chunks;
  for (int shard = 0; shard < num_shards_; ++shard) {
    const int client_index = shard_to_client_index_.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
      const std::size_t end = std::min(
          offset + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
          shard_keys[static_cast<std::size_t>(shard)].size());
      ShardChunk chunk;
      chunk.shard_id     = shard;
      chunk.client_index = client_index;
      chunk.keys.assign(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      chunk.positions.assign(
          shard_positions[static_cast<std::size_t>(shard)].begin() + offset,
          shard_positions[static_cast<std::size_t>(shard)].begin() + end);
      chunks.push_back(std::move(chunk));
    }
  }
  return chunks;
}

bool RDMAPSClientAdapter::FinalizeBatchIfNeeded(BatchRequest* batch) {
  if (batch == nullptr) {
    return false;
  }
  if (batch->assembled) {
    return batch->status_code ==
           static_cast<std::int32_t>(petps::RpcStatus::kOk);
  }

  batch->status_code = static_cast<std::int32_t>(petps::RpcStatus::kOk);
  for (const auto& pending : batch->shard_rpcs) {
    const auto* status_word = petps::FixedSlotStatusWord(
        pending.recv_buffer, pending.key_count, FLAGS_value_size);
    if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
      batch->status_code = *status_word;
      break;
    }
  }

  const int embedding_dim = FLAGS_value_size / sizeof(float);
  if (batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    for (const auto& pending : batch->shard_rpcs) {
      const float* shard_values =
          static_cast<const float*>(pending.recv_buffer);
      for (std::size_t i = 0; i < pending.original_positions.size(); ++i) {
        std::memcpy(
            batch->user_buffer + pending.original_positions[i] * embedding_dim,
            shard_values + i * embedding_dim,
            FLAGS_value_size);
      }
    }
  }

  auto* batch_status_word = reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(batch->user_buffer) +
      batch->total_key_count * static_cast<std::size_t>(FLAGS_value_size));
  *batch_status_word = batch->status_code;
  batch->assembled   = true;
  return batch->status_code == static_cast<std::int32_t>(petps::RpcStatus::kOk);
}

void InitializeRdmaProcessRuntime() {
  static std::once_flag init_once;
  std::call_once(init_once, []() {
    std::vector<std::string> argv_strings = ReadProcessArgv();
    if (argv_strings.empty()) {
      argv_strings.emplace_back("recstore_rdma_client");
    }

    std::vector<char*> argv_storage;
    argv_storage.reserve(argv_strings.size() + 1);
    for (auto& arg : argv_strings) {
      argv_storage.push_back(arg.data());
    }
    argv_storage.push_back(nullptr);

    int argc    = static_cast<int>(argv_strings.size());
    char** argv = argv_storage.data();
    folly::init(&argc, &argv);
  });
}

RDMAPSClientAdapter::RDMAPSClientAdapter(json config)
    : BasePSClient(config), config_(std::move(config)) {}

void RDMAPSClientAdapter::EnsureClientInitialized() {
  std::lock_guard<std::mutex> guard(init_mu_);
  if (initialized_) {
    return;
  }

  const json cache_ps_cfg =
      config_.contains("cache_ps") ? config_["cache_ps"] : json::object();
  const json client_cfg =
      config_.contains("client") ? config_["client"] : json::object();
  const json dist_cfg = ResolveFrameworkDistributedClientConfig(config_);

  num_shards_  = dist_cfg.value("num_shards", 1);
  hash_method_ = dist_cfg.value("hash_method", "city_hash");
  if (FLAGS_global_id < num_shards_) {
    FLAGS_num_server_processes = num_shards_;
    FLAGS_num_client_processes = 1;
    FLAGS_global_id            = num_shards_;
  } else if (FLAGS_num_server_processes != num_shards_) {
    throw std::runtime_error(
        "RDMA num_server_processes must match distributed_client.num_shards");
  }
  FLAGS_value_size =
      cache_ps_cfg.contains("base_kv_config")
          ? ValueSizeHintFromBaseKvConfig(
                cache_ps_cfg["base_kv_config"], FLAGS_value_size)
          : FLAGS_value_size;
  FLAGS_max_kv_num_per_request =
      dist_cfg.value("max_keys_per_request", FLAGS_max_kv_num_per_request);
  if (const char* mode = std::getenv("RECSTORE_RDMA_TRANSPORT_MODE")) {
    FLAGS_rdma_transport_mode = mode;
  }

  shard_clients_.clear();
  shard_to_client_index_.clear();
  client_ = nullptr;

  if (num_shards_ <= 1) {
    shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
        client_cfg.value("host", std::string("127.0.0.1")),
        client_cfg.value("port", 25000),
        client_cfg.value("shard", 0)));
    client_                   = shard_clients_.front().get();
    shard_to_client_index_[0] = 0;
  } else {
    const auto servers_it = dist_cfg.find("servers");
    if (servers_it == dist_cfg.end() || !servers_it->is_array() ||
        servers_it->empty()) {
      throw std::runtime_error(
          "RDMA distributed_client.servers must be provided for multi-shard "
          "configuration");
    }

    CHECK_EQ(static_cast<int>(servers_it->size()), num_shards_)
        << "RDMA distributed_client.servers size must equal num_shards";
    for (const auto& server : *servers_it) {
      const int shard = server.value("shard", -1);
      if (shard < 0) {
        throw std::runtime_error(
            "RDMA distributed_client.servers[].shard must be explicit");
      }
      shard_clients_.push_back(std::make_unique<petps::PetPSClient>(
          server.value("host", std::string("127.0.0.1")),
          server.value("port", 25000),
          shard));
      shard_to_client_index_[shard] =
          static_cast<int>(shard_clients_.size() - 1);
    }
  }

  initialized_ = true;
}

void RDMAPSClientAdapter::EnsureThreadInitialized() {
  EnsureClientInitialized();
  const std::thread::id tid = std::this_thread::get_id();
  std::lock_guard<std::mutex> guard(thread_init_mu_);
  if (initialized_threads_.find(tid) != initialized_threads_.end()) {
    return;
  }

  if (num_shards_ <= 1) {
    if (client_ != nullptr) {
      client_->InitThread();
    }
  } else {
    for (auto& shard_client : shard_clients_) {
      shard_client->InitThread();
    }
  }

  initialized_threads_.insert(tid);
}

void RDMAPSClientAdapter::EnsureTableReady(const std::string& table_name,
                                           int64_t embedding_dim) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = tables_.find(table_name);
  if (it == tables_.end()) {
    throw std::runtime_error("RDMA table is not initialized: " + table_name);
  }
  if (static_cast<int64_t>(it->second.config.embedding_dim) != embedding_dim) {
    throw std::runtime_error(
        "RDMA embedding dimension mismatch for table " + table_name);
  }
}

int64_t RDMAPSClientAdapter::DefaultEmbeddingDimOrThrow() const {
  if (tables_.empty()) {
    throw std::runtime_error(
        "RDMA table metadata is empty; call InitEmbeddingTable first");
  }
  return static_cast<int64_t>(tables_.begin()->second.config.embedding_dim);
}

RDMAPSClientAdapter::PrefetchState
RDMAPSClientAdapter::GetPrefetchState(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  const auto it = prefetches_.find(prefetch_id);
  if (it == prefetches_.end()) {
    throw std::runtime_error(
        "Unknown RDMA prefetch id: " + std::to_string(prefetch_id));
  }
  return it->second;
}

void RDMAPSClientAdapter::MarkPrefetchConsumed(uint64_t prefetch_id) {
  std::lock_guard<std::mutex> guard(state_mu_);
  prefetches_.erase(prefetch_id);
}

bool RDMAPSClientAdapter::QueryRPCFinished(int rpc_id) {
  if (num_shards_ <= 1) {
    return client_ != nullptr ? client_->QueryRPCFinished(rpc_id) : true;
  }

  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    if (!shard_clients_[static_cast<std::size_t>(pending.client_index)]
             ->QueryRPCFinished(pending.rpc_id)) {
      return false;
    }
  }

  return FinalizeBatchIfNeeded(&it->second);
}

void RDMAPSClientAdapter::WaitRPCFinish(int rpc_id) {
  if (num_shards_ <= 1) {
    if (client_ != nullptr) {
      client_->WaitRPCFinish(rpc_id);
    }
    return;
  }

  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    shard_clients_[static_cast<std::size_t>(pending.client_index)]
        ->WaitRPCFinish(pending.rpc_id);
  }

  FinalizeBatchIfNeeded(&it->second);
}

void RDMAPSClientAdapter::RevokeRPCResource(int rpc_id) {
  if (num_shards_ <= 1) {
    if (client_ != nullptr) {
      client_->RevokeRPCResource(rpc_id);
    }
    return;
  }

  std::lock_guard<std::mutex> guard(batches_mu_);
  auto it = batches_.find(rpc_id);
  CHECK(it != batches_.end());

  for (const auto& pending : it->second.shard_rpcs) {
    shard_clients_[static_cast<std::size_t>(pending.client_index)]
        ->RevokeRPCResource(pending.rpc_id);
  }

  batches_.erase(it);
}

int RDMAPSClientAdapter::SubmitGetParameter(
    base::ConstArray<uint64_t> keys,
    float* values,
    bool isAsync,
    int async_req_id) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    auto* status =
        reinterpret_cast<std::int32_t*>(reinterpret_cast<char*>(values));
    *status = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    return 0;
  }

  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    return client_->GetParameter(keys, values, isAsync, async_req_id);
  }

  BatchRequest batch;
  batch.user_buffer     = values;
  batch.total_key_count = keys.Size();
  auto* batch_status_word =
      petps::FixedSlotStatusWord(values, keys.Size(), FLAGS_value_size);
  *batch_status_word = static_cast<std::int32_t>(petps::RpcStatus::kPending);

  for (const auto& chunk : BuildChunks(keys)) {
    BaseParameterClient* client = shard_clients_[chunk.client_index].get();
    void* recv                  = client->GetReceiveBuffer(
        chunk.keys.size() * static_cast<std::size_t>(FLAGS_value_size) +
        sizeof(std::int32_t));
    const int rpc_id = client->GetParameter(
        base::ConstArray<uint64_t>(chunk.keys),
        static_cast<float*>(recv),
        isAsync,
        async_req_id);
    batch.shard_rpcs.push_back(PendingShardRpc{
        chunk.shard_id,
        chunk.client_index,
        rpc_id,
        chunk.positions,
        recv,
        chunk.keys.size(),
    });
  }

  std::uint64_t batch_id = 0;
  {
    std::lock_guard<std::mutex> guard(batches_mu_);
    batch_id = batch_rpc_id_acc_++;
    if (batch_id >
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error(
          "rdma batch rpc id overflow int range: " + std::to_string(batch_id));
    }
    batches_[batch_id] = std::move(batch);
  }
  if (!isAsync) {
    WaitRPCFinish(static_cast<int>(batch_id));
  }
  return static_cast<int>(batch_id);
}

int RDMAPSClientAdapter::GetParameter(const base::ConstArray<uint64_t>& keys,
                                      float* values) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    return 0;
  }

  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  float* recv = nullptr;
  if (num_shards_ > 1) {
    if (shard_clients_.empty()) {
      return -1;
    }
    recv = static_cast<float*>(
        shard_clients_.front()->GetReceiveBuffer(response_bytes));
    std::memset(recv, 0, response_bytes);
  } else {
    recv = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
    std::memset(recv, 0, response_bytes);
  }

  const int rpc_id = SubmitGetParameter(keys, recv, false, 0);
  WaitRPCFinish(rpc_id);
  const auto* status_word =
      petps::FixedSlotStatusWord(recv, keys.Size(), FLAGS_value_size);
  if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }

  std::memcpy(
      values, recv, keys.Size() * static_cast<std::size_t>(FLAGS_value_size));
  RevokeRPCResource(rpc_id);
  return 0;
}

int RDMAPSClientAdapter::PutParameter(
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>& values) {
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    return client_->PutParameter(keys.ToVector(), values);
  }
  if (keys.Size() != values.size()) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_values(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i]);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_values[static_cast<std::size_t>(shard)].push_back(values[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    const int client_index = shard_to_client_index_.at(shard);
    for (std::size_t offset = 0;
         offset < shard_keys[static_cast<std::size_t>(shard)].size();
         offset += static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
      const std::size_t end = std::min(
          offset + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
          shard_keys[static_cast<std::size_t>(shard)].size());
      std::vector<uint64_t> key_slice(
          shard_keys[static_cast<std::size_t>(shard)].begin() + offset,
          shard_keys[static_cast<std::size_t>(shard)].begin() + end);
      std::vector<std::vector<float>> value_slice(
          shard_values[static_cast<std::size_t>(shard)].begin() + offset,
          shard_values[static_cast<std::size_t>(shard)].begin() + end);
      int rc =
          shard_clients_[static_cast<std::size_t>(client_index)]->PutParameter(
              key_slice, value_slice);
      if (rc != 0) {
        return rc;
      }
    }
  }
  return 0;
}

int RDMAPSClientAdapter::UpdateParameter(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const std::vector<std::vector<float>>* grads) {
  if (grads == nullptr) {
    return -1;
  }
  if (grads->empty()) {
    return 0;
  }
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    return client_->UpdateParameter(table_name, keys, grads);
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }
  if (keys.Size() == 0) {
    return 0;
  }

  std::vector<std::vector<uint64_t>> shard_keys(num_shards_);
  std::vector<std::vector<std::vector<float>>> shard_grads(num_shards_);

  for (std::size_t i = 0; i < keys.Size(); ++i) {
    const int shard = PartitionKey(keys[i]);
    shard_keys[static_cast<std::size_t>(shard)].push_back(keys[i]);
    shard_grads[static_cast<std::size_t>(shard)].push_back((*grads)[i]);
  }

  for (int shard = 0; shard < num_shards_; ++shard) {
    if (shard_keys[static_cast<std::size_t>(shard)].empty()) {
      continue;
    }
    const int client_index = shard_to_client_index_.at(shard);
    const int rc =
        shard_clients_[static_cast<std::size_t>(client_index)]->UpdateParameter(
            table_name,
            base::ConstArray<uint64_t>(
                shard_keys[static_cast<std::size_t>(shard)]),
            &shard_grads[static_cast<std::size_t>(shard)]);
    if (rc != 0) {
      return rc;
    }
  }
  return 0;
}

int RDMAPSClientAdapter::UpdateParameterFlat(
    const std::string& table_name,
    const base::ConstArray<uint64_t>& keys,
    const float* grads,
    int64_t num_rows,
    int64_t embedding_dim) {
  EnsureTableReady(table_name, embedding_dim);
  if (num_rows == 0) {
    return 0;
  }
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != static_cast<std::size_t>(num_rows)) {
    return -1;
  }
  std::vector<std::vector<float>> updated;
  updated.reserve(static_cast<std::size_t>(num_rows));
  for (int64_t row = 0; row < num_rows; ++row) {
    std::vector<float> values(static_cast<std::size_t>(embedding_dim), 0.0f);
    for (int64_t col = 0; col < embedding_dim; ++col) {
      const std::size_t idx =
          static_cast<std::size_t>(row * embedding_dim + col);
      values[static_cast<std::size_t>(col)] = grads[idx];
    }
    updated.push_back(std::move(values));
  }

  return UpdateParameter(table_name, keys, &updated);
}

int RDMAPSClientAdapter::InitEmbeddingTable(
    const std::string& table_name, const EmbeddingTableConfig& config) {
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      return -1;
    }
    const int init_rc = client_->InitEmbeddingTable(
        table_name, config.num_embeddings, config.embedding_dim);
    if (init_rc != 0) {
      return init_rc;
    }
  } else {
    for (auto& shard_client : shard_clients_) {
      const int rc = shard_client->InitEmbeddingTable(
          table_name, config.num_embeddings, config.embedding_dim);
      if (rc != 0) {
        return rc;
      }
    }
  }

  std::lock_guard<std::mutex> guard(state_mu_);
  const auto [it, inserted] = tables_.emplace(table_name, TableState{config});
  if (!inserted) {
    if (it->second.config.embedding_dim != config.embedding_dim ||
        it->second.config.num_embeddings != config.num_embeddings) {
      return -1;
    }
  }
  return 0;
}

int RDMAPSClientAdapter::AsyncGetParameter(const base::ConstArray<uint64_t>&,
                                           float*) {
  throw std::runtime_error(
      "RDMA adapter AsyncGetParameter not implemented yet");
}

void RDMAPSClientAdapter::Command(PSCommand) {
  EnsureThreadInitialized();
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      throw std::runtime_error("RDMA adapter has no initialized client");
    }
    client_->Barrier("rdma_command", 0);
    return;
  }
  if (shard_clients_.empty()) {
    throw std::runtime_error("RDMA adapter has no initialized clients");
  }
  shard_clients_.front()->Barrier("rdma_command", 0);
}

uint64_t
RDMAPSClientAdapter::PrefetchParameter(const base::ConstArray<uint64_t>& keys) {
  EnsureThreadInitialized();
  if (keys.Size() == 0) {
    throw std::invalid_argument("RDMA prefetch requires at least one key");
  }

  const int64_t embedding_dim = DefaultEmbeddingDimOrThrow();
  const std::size_t response_bytes =
      petps::FixedSlotResponseBytes(keys.Size(), FLAGS_value_size);
  float* buffer = nullptr;
  if (num_shards_ <= 1) {
    if (client_ == nullptr) {
      throw std::runtime_error("RDMA adapter has no initialized client");
    }
    buffer = static_cast<float*>(client_->GetReceiveBuffer(response_bytes));
  } else {
    if (shard_clients_.empty()) {
      throw std::runtime_error("RDMA adapter has no initialized clients");
    }
    buffer = static_cast<float*>(
        shard_clients_.front()->GetReceiveBuffer(response_bytes));
  }
  std::memset(buffer, 0, response_bytes);

  const int rpc_id = SubmitGetParameter(keys, buffer, true, 0);

  std::lock_guard<std::mutex> guard(state_mu_);
  const uint64_t prefetch_id = next_prefetch_id_++;
  prefetches_.emplace(
      prefetch_id,
      PrefetchState{
          buffer,
          rpc_id,
          static_cast<int64_t>(keys.Size()),
          embedding_dim,
      });
  return prefetch_id;
}

bool RDMAPSClientAdapter::IsPrefetchDone(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  const PrefetchState state = GetPrefetchState(prefetch_id);
  return QueryRPCFinished(state.rpc_id);
}

void RDMAPSClientAdapter::WaitForPrefetch(uint64_t prefetch_id) {
  EnsureThreadInitialized();
  const PrefetchState state = GetPrefetchState(prefetch_id);
  try {
    WaitRPCFinish(state.rpc_id);
  } catch (...) {
    RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    throw;
  }
}

bool RDMAPSClientAdapter::GetPrefetchResult(
    uint64_t prefetch_id, std::vector<std::vector<float>>* values) {
  if (values == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  std::vector<float> flat;
  int64_t num_rows = 0;
  if (!GetPrefetchResultFlat(
          prefetch_id, &flat, &num_rows, state.embedding_dim)) {
    return false;
  }

  petps::CopyFlatRowsToVectors(
      flat.data(),
      static_cast<std::size_t>(num_rows),
      static_cast<std::size_t>(state.embedding_dim),
      values);
  return true;
}

bool RDMAPSClientAdapter::GetPrefetchResultFlat(
    uint64_t prefetch_id,
    std::vector<float>* values,
    int64_t* num_rows,
    int64_t embedding_dim) {
  if (values == nullptr || num_rows == nullptr) {
    return false;
  }

  const PrefetchState state = GetPrefetchState(prefetch_id);
  if (embedding_dim != state.embedding_dim) {
    return false;
  }

  WaitForPrefetch(prefetch_id);
  const auto* status_word = petps::FixedSlotStatusWord(
      state.buffer,
      static_cast<std::size_t>(state.key_count),
      FLAGS_value_size);
  if (*status_word != static_cast<std::int32_t>(petps::RpcStatus::kOk)) {
    RevokeRPCResource(state.rpc_id);
    MarkPrefetchConsumed(prefetch_id);
    return false;
  }

  const std::size_t value_count =
      static_cast<std::size_t>(state.key_count) *
      static_cast<std::size_t>(state.embedding_dim);
  values->assign(state.buffer, state.buffer + value_count);
  *num_rows = state.key_count;
  RevokeRPCResource(state.rpc_id);
  MarkPrefetchConsumed(prefetch_id);
  return true;
}

} // namespace recstore
