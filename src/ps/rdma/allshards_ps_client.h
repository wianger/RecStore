#pragma once

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/log.h"
#include "ps/rdma/base_client.h"
#include "ps/rdma/rdma_status.h"

class AllShardsParameterClientWrapper : public BaseParameterClient {
public:
  explicit AllShardsParameterClientWrapper(
      const std::vector<BaseParameterClient*>& clients,
      int num_shards,
      const std::string& hash_method    = "city_hash",
      const std::vector<int>& shard_ids = {});

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;

  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  void InitThread() override;
  void Barrier(const std::string& ss, int k) override;
  void* GetReceiveBuffer(size_t size) override;
  bool QueryRPCFinished(int rpc_id) override;
  void WaitRPCFinish(int rpc_id) override;
  void RevokeRPCResource(int rpc_id) override;
  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;
  int InitEmbeddingTable(const std::string& table_name,
                         std::uint64_t num_embeddings,
                         std::uint64_t embedding_dim) override;
  int UpdateParameter(const std::string& table_name,
                      base::ConstArray<uint64_t> keys,
                      const std::vector<std::vector<float>>* grads) override;

private:
  struct PendingShardRpc {
    int shard_id     = 0;  // Logical shard this RPC belongs to.
    int client_index = 0;  // Underlying client selected for this shard.
    int rpc_id       = -1; // RPC id returned by the shard-local client.
    std::vector<std::size_t>
        original_positions;          // Positions in the caller's batch.
    void* recv_buffer     = nullptr; // Caller-visible response buffer.
    std::size_t key_count = 0;       // Number of keys in this shard chunk.
  };

  struct BatchRequest {
    float* user_buffer = nullptr; // Final output buffer owned by caller.
    bool assembled     = false;   // True once all shard RPCs have been merged.
    std::size_t total_key_count = 0; // Total keys across all shards.
    std::int32_t status_code =
        static_cast<std::int32_t>(petps::RpcStatus::kPending);
    std::vector<PendingShardRpc> shard_rpcs; // One pending RPC per shard chunk.
  };

  struct ShardChunk {
    int shard_id     = 0;               // Routed shard id.
    int client_index = 0;               // Client that serves this shard.
    std::vector<uint64_t> keys;         // Keys assigned to this shard chunk.
    std::vector<std::size_t> positions; // Original positions in caller input.
  };

  int PartitionKey(uint64_t key) const;
  std::vector<ShardChunk> BuildChunks(base::ConstArray<uint64_t> keys) const;
  bool FinalizeBatchIfNeeded(BatchRequest* batch);

  std::vector<BaseParameterClient*> clients_; // Shard-local clients.
  int num_shards_; // Number of logical shards in the distributed layout.
  std::string hash_method_; // Hash method used to map keys to shards.
  std::unordered_map<int, int>
      shard_to_client_index_;          // Explicit shard -> client map.
  std::uint64_t batch_rpc_id_acc_ = 1; // Wrapper-local batch handle generator.
  mutable std::mutex batches_mu_; // Guards batch assembly and completion state.
  std::unordered_map<std::uint64_t, BatchRequest>
      batches_; // Outstanding wrapper batches.
};
