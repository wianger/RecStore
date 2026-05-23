#pragma once
#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"
#include "base_client.h"
#include "rdma_protocol.h"
#include "rdma_status.h"
#include "third_party/Mayfly-main/include/DSM.h"

namespace petps {
class PetPSClient : public BaseParameterClient {
public:
  explicit PetPSClient(const std::string& host, int port, int shard)
      : BaseParameterClient(host, port, shard) {
    Init();
  }

  ~PetPSClient() {}

  void Barrier(const std::string& ss, int k) override { dsm_->barrier(ss, k); }

  void InitThread() override {
    LOG(INFO) << "dsm_->registerThread()";
    dsm_->registerThread();
    WaitForServerReady();
    serverThreadIdsRoutedTo_ = GetServerThreadIDs();
  }

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>>* values) override;

  // this interface assume all keys with the same embedding dimension
  int GetParameter(base::ConstArray<uint64_t> keys,
                   float* values,
                   bool isAsync,
                   int async_req_id = 0) override;

  std::size_t ResponseBufferBytes(std::size_t key_count) const;

  void* GetReceiveBuffer(size_t size) override;
  void* GetSendBuffer(size_t size);

  inline int shard() const { return shard_; }

  bool QueryRPCFinished(int rpc_id) override;

  void WaitRPCFinish(int rpc_id) override;

  void RevokeRPCResource(int rpc_id) override;

  int PutParameter(const std::vector<uint64_t>& keys,
                   const std::vector<std::vector<float>>& values) override;

  int FakePutParameter(base::ConstArray<uint64_t> keys, float* values) override;

private:
  void WaitForServerReady();
  std::vector<int> GetServerThreadIDs();
  int SelectServerThreadID() const;
  std::atomic<int32_t>* GetPollSlot(uint64_t rpc_id) const;
  void Init();
  DSM* dsm_;

  mutable std::mutex rpc_mu_;
  mutable std::mutex put_mu_;
  std::atomic<uint64_t> rpcIDAcc_{0};
  mutable std::atomic<uint32_t> round_robin_{0};

  std::vector<int> serverThreadIdsRoutedTo_;
  std::unordered_map<uint64_t, std::atomic<int32_t>*> rpcId2PollMap_;
};

FACTORY_REGISTER(BaseParameterClient,
                 PetPSClient,
                 PetPSClient,
                 const std::string&,
                 int,
                 int);
} // namespace petps
