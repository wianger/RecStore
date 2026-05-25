#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"
#include "ps/rdma/base_client.h"
#include "ps/rdma/rc_transport.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"

namespace petps {

class PetPSClient : public BaseParameterClient {
public:
  explicit PetPSClient(const std::string& host, int port, int shard);
  ~PetPSClient() override;

  void Barrier(const std::string& ss, int k) override;
  void InitThread() override;

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;
  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  std::size_t ResponseBufferBytes(std::size_t key_count) const;

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
  int FakePutParameter(base::ConstArray<uint64_t> keys, float* values) override;

private:
  struct QpContext {
    RcClientQpView view;            // Local slot view for one client-side lane.
    std::uint64_t next_seq = 1;     // Monotonic request sequence for this lane.
    bool busy              = false; // True while one RPC is in flight.
  };

  struct PendingRpc {
    int qp_index               = -1;      // Lane used for this pending RPC.
    std::uint64_t seq          = 0;       // Sequence number written into slot.
    float* recv_buffer         = nullptr; // User-visible response buffer.
    std::size_t key_count      = 0;       // Number of keys in this RPC.
    std::size_t response_bytes = 0;       // Dense response payload bytes.
  };

  void InitializeTransport();
  int AcquireIdleQp();
  void FillGetDescriptor(RequestDescriptor* descriptor,
                         std::uint64_t seq,
                         std::size_t key_count,
                         std::size_t response_bytes,
                         const RcClientQpView& view) const;
  void FillPutDescriptor(RequestDescriptor* descriptor,
                         std::uint64_t seq,
                         std::size_t key_count,
                         std::size_t payload_bytes,
                         const RcClientQpView& view) const;
  void FillUpdateDescriptor(
      RequestDescriptor* descriptor,
      std::uint64_t seq,
      std::size_t key_count,
      std::size_t payload_bytes,
      const std::string& table_name,
      const RcClientQpView& view) const;
  void FillInitTableDescriptor(RequestDescriptor* descriptor,
                               std::uint64_t seq,
                               const std::string& table_name,
                               const RcClientQpView& view) const;
  int SubmitRpcLocked(
      int qp_index,
      RequestDescriptor* descriptor,
      const void* payload,
      std::size_t payload_bytes,
      float* recv_buffer,
      std::size_t key_count,
      std::size_t response_bytes,
      bool is_async);

  std::string namespace_token_; // Shared-memory namespace token.
  int client_id_ = -1;          // Logical client id derived from global id.
  RcTransportConfig config_;    // Transport slot sizing and shard config.
  std::unique_ptr<RcShardClientTransport> transport_; // Slot transport owner.
  std::vector<QpContext> qps_; // One context per client-side lane.
  std::vector<std::vector<char>>
      receive_buffers_; // Heap-backed response buffers.
  std::unordered_map<int, PendingRpc> pending_rpcs_; // In-flight RPC table.
  std::mutex mu_; // Guards transport setup and pending RPC state.
  std::atomic<int> next_rpc_id_{1}; // Monotonic RPC handle generator.
  bool thread_initialized_ = false; // Set after InitThread has run.
};

FACTORY_REGISTER(BaseParameterClient,
                 PetPSClient,
                 PetPSClient,
                 const std::string&,
                 int,
                 int);

} // namespace petps
