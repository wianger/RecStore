#include "ps/rdma/petps_client.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <thread>

#include <folly/portability/GFlags.h>

#include "ps/rdma/rc_options.h"

DECLARE_int32(global_id);
DECLARE_int32(num_server_processes);
DECLARE_int32(num_client_processes);
DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

namespace petps {
namespace {

std::string NamespaceToken() {
  if (!FLAGS_rdma_rc_namespace.empty()) {
    return FLAGS_rdma_rc_namespace;
  }
  if (const char* env = std::getenv("RECSTORE_MEMCACHED_NAMESPACE")) {
    return env;
  }
  return "default";
}

std::size_t ComputeMaxGetKeysPerRpc() {
  return GetKeysPerRpcByResponseBudget(
      static_cast<std::size_t>(FLAGS_value_size),
      static_cast<std::size_t>(FLAGS_rdma_rc_mtu_bytes),
      static_cast<std::size_t>(FLAGS_rdma_rc_target_response_mtu));
}

std::int32_t WaitStatus(const StatusWord* status, std::uint64_t seq) {
  const auto start = std::chrono::steady_clock::now();
  while (!StatusWordDone(*status, seq)) {
    std::this_thread::yield();
    if (FLAGS_rdma_wait_timeout_ms > 0) {
      const auto elapsed_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (elapsed_ms > FLAGS_rdma_wait_timeout_ms) {
        throw std::runtime_error("RC write RPC wait timeout");
      }
    }
  }
  return status->status;
}

} // namespace

PetPSClient::PetPSClient(const std::string& host, int port, int shard)
    : BaseParameterClient(host, port, shard),
      namespace_token_(NamespaceToken()) {}

PetPSClient::~PetPSClient() = default;

void PetPSClient::Barrier(const std::string&, int) {}

void PetPSClient::InitializeTransport() {
  if (transport_ != nullptr) {
    return;
  }
  client_id_ = FLAGS_global_id - FLAGS_num_server_processes;
  if (client_id_ < 0) {
    throw std::runtime_error("invalid RC write client_id from global_id");
  }
  config_.shard_id                 = shard_;
  config_.client_id                = client_id_;
  config_.num_clients              = FLAGS_num_client_processes;
  config_.qps_per_client_per_shard = FLAGS_rdma_rc_qps_per_client_per_shard;
  config_.request_slot_bytes =
      static_cast<std::size_t>(FLAGS_rdma_rc_request_slot_bytes);
  config_.response_slot_bytes =
      static_cast<std::size_t>(FLAGS_rdma_rc_response_slot_bytes);
  config_.namespace_token = namespace_token_;

  transport_ = std::make_unique<RcShardClientTransport>(config_);
  qps_.clear();
  qps_.reserve(static_cast<std::size_t>(config_.qps_per_client_per_shard));
  for (int qp = 0; qp < config_.qps_per_client_per_shard; ++qp) {
    qps_.push_back(QpContext{transport_->OpenQp(qp), 1, false});
  }
}

void PetPSClient::InitThread() {
  std::lock_guard<std::mutex> guard(mu_);
  InitializeTransport();
  thread_initialized_ = true;
}

std::size_t PetPSClient::ResponseBufferBytes(std::size_t key_count) const {
  return GetResponseBytes(
             key_count, static_cast<std::size_t>(FLAGS_value_size)) +
         sizeof(std::int32_t);
}

void* PetPSClient::GetReceiveBuffer(size_t size) {
  std::lock_guard<std::mutex> guard(mu_);
  receive_buffers_.emplace_back(size, 0);
  return receive_buffers_.back().data();
}

int PetPSClient::AcquireIdleQp() {
  for (std::size_t i = 0; i < qps_.size(); ++i) {
    if (!qps_[i].busy) {
      qps_[i].busy = true;
      return static_cast<int>(i);
    }
  }
  throw std::runtime_error("no idle RC write QP available");
}

void PetPSClient::FillGetDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t response_bytes,
    const RcClientQpView& view) const {
  *descriptor            = RequestDescriptor{};
  descriptor->op         = static_cast<std::uint16_t>(RcOp::kGet);
  descriptor->seq        = seq;
  descriptor->shard_id   = static_cast<std::uint32_t>(shard_);
  descriptor->client_id  = static_cast<std::uint32_t>(client_id_);
  descriptor->qp_index   = static_cast<std::uint32_t>(view.qp_index);
  descriptor->key_count  = static_cast<std::uint32_t>(key_count);
  descriptor->value_size = static_cast<std::uint32_t>(FLAGS_value_size);
  descriptor->embedding_dim =
      static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float));
  descriptor->payload_offset =
      static_cast<std::uint32_t>(Align64(sizeof(RequestDescriptor)));
  descriptor->payload_bytes =
      static_cast<std::uint32_t>(GetRequestBytes(key_count));
  descriptor->response_bytes = static_cast<std::uint32_t>(response_bytes);
  descriptor->client_response_addr =
      reinterpret_cast<std::uint64_t>(view.response_payload);
  descriptor->client_status_addr = reinterpret_cast<std::uint64_t>(view.status);
}

void PetPSClient::FillPutDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t payload_bytes,
    const RcClientQpView& view) const {
  *descriptor            = RequestDescriptor{};
  descriptor->op         = static_cast<std::uint16_t>(RcOp::kPut);
  descriptor->seq        = seq;
  descriptor->shard_id   = static_cast<std::uint32_t>(shard_);
  descriptor->client_id  = static_cast<std::uint32_t>(client_id_);
  descriptor->qp_index   = static_cast<std::uint32_t>(view.qp_index);
  descriptor->key_count  = static_cast<std::uint32_t>(key_count);
  descriptor->value_size = static_cast<std::uint32_t>(FLAGS_value_size);
  descriptor->embedding_dim =
      static_cast<std::uint32_t>(FLAGS_value_size / sizeof(float));
  descriptor->payload_offset =
      static_cast<std::uint32_t>(Align64(sizeof(RequestDescriptor)));
  descriptor->payload_bytes  = static_cast<std::uint32_t>(payload_bytes);
  descriptor->response_bytes = 0;
  descriptor->client_response_addr =
      reinterpret_cast<std::uint64_t>(view.response_payload);
  descriptor->client_status_addr = reinterpret_cast<std::uint64_t>(view.status);
}

void PetPSClient::FillUpdateDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    std::size_t key_count,
    std::size_t payload_bytes,
    const std::string& table_name,
    const RcClientQpView& view) const {
  FillPutDescriptor(descriptor, seq, key_count, payload_bytes, view);
  descriptor->op = static_cast<std::uint16_t>(RcOp::kUpdate);
  if (!CopyTableName(table_name, &descriptor->table_name)) {
    throw std::runtime_error("UPDATE table name too long");
  }
}

void PetPSClient::FillInitTableDescriptor(
    RequestDescriptor* descriptor,
    std::uint64_t seq,
    const std::string& table_name,
    const RcClientQpView& view) const {
  FillPutDescriptor(
      descriptor, seq, /*key_count=*/0, InitTablePayloadBytes(), view);
  descriptor->op = static_cast<std::uint16_t>(RcOp::kInitTable);
  if (!CopyTableName(table_name, &descriptor->table_name)) {
    throw std::runtime_error("INIT table name too long");
  }
}

int PetPSClient::SubmitRpcLocked(
    int qp_index,
    RequestDescriptor* descriptor,
    const void* payload,
    std::size_t payload_bytes,
    float* recv_buffer,
    std::size_t key_count,
    std::size_t response_bytes,
    bool is_async) {
  auto& qp = qps_.at(static_cast<std::size_t>(qp_index));
  ResetStatusWord(qp.view.status, descriptor->seq);
  transport_->SubmitRequest(qp.view, *descriptor, payload, payload_bytes);
  VLOG(1) << "component=rdma_rc_client event=submit shard=" << shard_
          << " client_id=" << client_id_ << " qp=" << qp_index
          << " slot=" << qp.view.slot_index << " seq=" << descriptor->seq
          << " op=" << descriptor->op << " key_count=" << key_count
          << " payload_bytes=" << payload_bytes
          << " response_bytes=" << response_bytes;

  const int rpc_id = next_rpc_id_.fetch_add(1);
  pending_rpcs_.emplace(
      rpc_id,
      PendingRpc{
          qp_index,
          descriptor->seq,
          recv_buffer,
          key_count,
          response_bytes,
      });
  if (!is_async) {
    WaitRPCFinish(rpc_id);
  }
  return rpc_id;
}

int PetPSClient::GetParameter(base::ConstArray<uint64_t> keys,
                              std::vector<std::vector<float>>* values) {
  values->clear();
  if (keys.Size() == 0) {
    return 0;
  }
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<float> flat(keys.Size() * embedding_dim + 1, 0.0f);
  const int rpc_id   = GetParameter(keys, flat.data(), false, 0);
  const auto* status = reinterpret_cast<const std::int32_t*>(
      reinterpret_cast<const char*>(flat.data()) +
      keys.Size() * static_cast<std::size_t>(FLAGS_value_size));
  if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
    RevokeRPCResource(rpc_id);
    return -1;
  }
  values->reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    values->emplace_back(flat.begin() + i * embedding_dim,
                         flat.begin() + (i + 1) * embedding_dim);
  }
  RevokeRPCResource(rpc_id);
  return 0;
}

int PetPSClient::GetParameter(
    base::ConstArray<uint64_t> keys, float* values, bool isAsync, int) {
  if (keys.Size() == 0) {
    auto* status =
        reinterpret_cast<std::int32_t*>(reinterpret_cast<char*>(values));
    *status = static_cast<std::int32_t>(RpcStatus::kOk);
    return 0;
  }
  int rpc_id = 0;
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!thread_initialized_) {
      throw std::runtime_error("PetPSClient::InitThread must be called first");
    }
    if (keys.Size() > ComputeMaxGetKeysPerRpc()) {
      throw std::runtime_error(
          "single-shard GET batch exceeds RC response budget");
    }

    const int qp_index = AcquireIdleQp();
    auto& qp           = qps_[static_cast<std::size_t>(qp_index)];
    RequestDescriptor descriptor;
    const std::size_t response_bytes = GetResponseBytes(
        keys.Size(), static_cast<std::size_t>(FLAGS_value_size));
    FillGetDescriptor(
        &descriptor, qp.next_seq++, keys.Size(), response_bytes, qp.view);
    if (descriptor.payload_bytes >
        PutPayloadBudget(config_.request_slot_bytes)) {
      qp.busy = false;
      throw std::runtime_error("GET request exceeds RC request slot");
    }
    rpc_id = SubmitRpcLocked(
        qp_index,
        &descriptor,
        keys.Data(),
        descriptor.payload_bytes,
        values,
        keys.Size(),
        response_bytes,
        true);
  }
  if (!isAsync) {
    WaitRPCFinish(rpc_id);
  }
  return rpc_id;
}

bool PetPSClient::QueryRPCFinished(int rpc_id) {
  std::lock_guard<std::mutex> guard(mu_);
  const auto it = pending_rpcs_.find(rpc_id);
  if (it == pending_rpcs_.end()) {
    return true;
  }
  const auto& qp = qps_.at(static_cast<std::size_t>(it->second.qp_index));
  return StatusWordDone(*qp.view.status, it->second.seq);
}

void PetPSClient::WaitRPCFinish(int rpc_id) {
  PendingRpc pending;
  {
    std::lock_guard<std::mutex> guard(mu_);
    const auto it = pending_rpcs_.find(rpc_id);
    if (it == pending_rpcs_.end()) {
      return;
    }
    pending = it->second;
  }

  auto& qp = qps_.at(static_cast<std::size_t>(pending.qp_index));
  const std::int32_t status_code = WaitStatus(qp.view.status, pending.seq);
  VLOG(1) << "component=rdma_rc_client event=done shard=" << shard_
          << " client_id=" << client_id_ << " qp=" << pending.qp_index
          << " seq=" << pending.seq << " status=" << status_code
          << " response_bytes=" << pending.response_bytes;
  if (pending.response_bytes > 0) {
    std::memcpy(
        pending.recv_buffer, qp.view.response_payload, pending.response_bytes);
  }
  auto* user_status = reinterpret_cast<std::int32_t*>(
      reinterpret_cast<char*>(pending.recv_buffer) +
      pending.key_count * static_cast<std::size_t>(FLAGS_value_size));
  *user_status = status_code;
}

void PetPSClient::RevokeRPCResource(int rpc_id) {
  std::lock_guard<std::mutex> guard(mu_);
  const auto it = pending_rpcs_.find(rpc_id);
  if (it == pending_rpcs_.end()) {
    return;
  }
  auto& qp = qps_.at(static_cast<std::size_t>(it->second.qp_index));
  transport_->ClearRequestSlot(qp.view);
  qp.busy = false;
  pending_rpcs_.erase(it);
}

int PetPSClient::PutParameter(const std::vector<uint64_t>& keys,
                              const std::vector<std::vector<float>>& values) {
  if (keys.size() != values.size()) {
    return -1;
  }
  if (keys.empty()) {
    return 0;
  }

  std::size_t begin = 0;
  while (begin < keys.size()) {
    std::size_t end =
        std::min(begin + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
                 keys.size());
    std::vector<std::uint64_t> key_slice(
        keys.begin() + begin, keys.begin() + end);
    std::vector<std::vector<float>> value_slice(
        values.begin() + begin, values.begin() + end);

    std::string payload;
    std::string error;
    const std::size_t payload_bytes =
        PutPayloadBytes(key_slice, value_slice, &payload, &error);
    if (payload_bytes == 0 && !key_slice.empty()) {
      throw std::runtime_error("RC PUT payload build failed: " + error);
    }

    float* recv = nullptr;
    int rpc_id  = 0;
    {
      std::lock_guard<std::mutex> guard(mu_);
      if (!thread_initialized_) {
        throw std::runtime_error(
            "PetPSClient::InitThread must be called first");
      }
      const int qp_index = AcquireIdleQp();
      auto& qp           = qps_[static_cast<std::size_t>(qp_index)];
      RequestDescriptor descriptor;
      FillPutDescriptor(
          &descriptor, qp.next_seq++, key_slice.size(), payload_bytes, qp.view);
      if (Align64(sizeof(RequestDescriptor)) + payload_bytes +
              Align64(sizeof(CommitWord)) >
          config_.request_slot_bytes) {
        qp.busy = false;
        throw std::runtime_error("PUT request exceeds RC request slot");
      }
      receive_buffers_.emplace_back(sizeof(std::int32_t), 0);
      recv   = reinterpret_cast<float*>(receive_buffers_.back().data());
      rpc_id = SubmitRpcLocked(
          qp_index,
          &descriptor,
          payload.data(),
          payload_bytes,
          recv,
          0,
          0,
          true);
    }
    WaitRPCFinish(rpc_id);
    const auto* status = reinterpret_cast<const std::int32_t*>(recv);
    RevokeRPCResource(rpc_id);
    if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      return -1;
    }
    begin = end;
  }

  return 0;
}

int PetPSClient::InitEmbeddingTable(const std::string& table_name,
                                    std::uint64_t num_embeddings,
                                    std::uint64_t embedding_dim) {
  const std::array<std::uint64_t, 2> payload_words = {
      num_embeddings,
      embedding_dim,
  };

  float* recv = nullptr;
  int rpc_id  = 0;
  {
    std::lock_guard<std::mutex> guard(mu_);
    if (!thread_initialized_) {
      throw std::runtime_error("PetPSClient::InitThread must be called first");
    }
    const int qp_index = AcquireIdleQp();
    auto& qp           = qps_[static_cast<std::size_t>(qp_index)];
    RequestDescriptor descriptor;
    FillInitTableDescriptor(&descriptor, qp.next_seq++, table_name, qp.view);
    if (Align64(sizeof(RequestDescriptor)) + descriptor.payload_bytes +
            Align64(sizeof(CommitWord)) >
        config_.request_slot_bytes) {
      qp.busy = false;
      throw std::runtime_error("INIT request exceeds RC request slot");
    }
    receive_buffers_.emplace_back(sizeof(std::int32_t), 0);
    recv   = reinterpret_cast<float*>(receive_buffers_.back().data());
    rpc_id = SubmitRpcLocked(
        qp_index,
        &descriptor,
        payload_words.data(),
        descriptor.payload_bytes,
        recv,
        0,
        0,
        true);
  }

  WaitRPCFinish(rpc_id);
  const auto* status = reinterpret_cast<const std::int32_t*>(recv);
  RevokeRPCResource(rpc_id);
  return (*status == static_cast<std::int32_t>(RpcStatus::kOk)) ? 0 : -1;
}

int PetPSClient::UpdateParameter(const std::string& table_name,
                                 base::ConstArray<uint64_t> keys,
                                 const std::vector<std::vector<float>>* grads) {
  if (keys.Size() == 0) {
    return 0;
  }
  if (grads == nullptr) {
    return -1;
  }
  if (keys.Size() != grads->size()) {
    return -1;
  }

  std::size_t begin            = 0;
  const std::size_t total_keys = static_cast<std::size_t>(keys.Size());
  while (begin < total_keys) {
    const std::size_t end =
        std::min(begin + static_cast<std::size_t>(FLAGS_max_kv_num_per_request),
                 total_keys);
    std::vector<std::uint64_t> key_slice(
        keys.Data() + begin, keys.Data() + end);
    std::vector<std::vector<float>> grad_slice(
        grads->begin() + begin, grads->begin() + end);

    std::string payload;
    std::string error;
    const std::size_t payload_bytes =
        UpdatePayloadBytes(key_slice, grad_slice, &payload, &error);
    if (payload_bytes == 0 && !key_slice.empty()) {
      throw std::runtime_error("RC UPDATE payload build failed: " + error);
    }

    float* recv = nullptr;
    int rpc_id  = 0;
    {
      std::lock_guard<std::mutex> guard(mu_);
      if (!thread_initialized_) {
        throw std::runtime_error(
            "PetPSClient::InitThread must be called first");
      }

      const int qp_index = AcquireIdleQp();
      auto& qp           = qps_[static_cast<std::size_t>(qp_index)];
      RequestDescriptor descriptor;
      FillUpdateDescriptor(
          &descriptor,
          qp.next_seq++,
          key_slice.size(),
          payload_bytes,
          table_name,
          qp.view);
      if (Align64(sizeof(RequestDescriptor)) + payload_bytes +
              Align64(sizeof(CommitWord)) >
          config_.request_slot_bytes) {
        qp.busy = false;
        throw std::runtime_error("UPDATE request exceeds RC request slot");
      }
      receive_buffers_.emplace_back(sizeof(std::int32_t), 0);
      recv   = reinterpret_cast<float*>(receive_buffers_.back().data());
      rpc_id = SubmitRpcLocked(
          qp_index,
          &descriptor,
          payload.data(),
          payload_bytes,
          recv,
          0,
          0,
          true);
    }

    WaitRPCFinish(rpc_id);
    const auto* status = reinterpret_cast<const std::int32_t*>(recv);
    RevokeRPCResource(rpc_id);
    if (*status != static_cast<std::int32_t>(RpcStatus::kOk)) {
      return -1;
    }
    begin = end;
  }

  return 0;
}

int PetPSClient::FakePutParameter(base::ConstArray<uint64_t> keys,
                                  float* values) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  std::vector<std::vector<float>> rows;
  rows.reserve(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    rows.emplace_back(
        values + i * embedding_dim, values + (i + 1) * embedding_dim);
  }
  return PutParameter(keys.ToVector(), rows);
}

} // namespace petps
