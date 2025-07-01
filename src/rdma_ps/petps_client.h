
#pragma once
#include <memory>
#include <string>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"
#include "base_client.h"
#include "third_party/Mayfly-main/include/DSM.h"

namespace petps {
class PetPSClient : public BaseParameterClient {
 public:
  explicit PetPSClient(const std::string &host, int port, int shard)
      : BaseParameterClient(host, port, shard) {
    Init();
  }

  ~PetPSClient() {}

  void Barrier(const std::string &ss, int k) override { dsm_->barrier(ss, k); }

  void InitThread() override {
    LOG(INFO) << "dsm_->registerThread()";
    dsm_->registerThread();
    serverThreadIdsRoutedTo_ = GetServerThreadIDs();
  }

  int GetParameter(base::ConstArray<uint64_t> keys,
                   std::vector<std::vector<float>> *values) override;

  // this interface assume all keys with the same embedding dimension
  int GetParameter(base::ConstArray<uint64_t> keys, float *values, bool isAsync,
                   int async_req_id = 0) override;

  void *GetReceiveBuffer(size_t size) override;

  inline int shard() const { return shard_; }

  bool QueryRPCFinished(int rpc_id) override;

  void WaitRPCFinish(int rpc_id) override;

  void RevokeRPCResource(int rpc_id) override;

  int PutParameter(const std::vector<uint64_t> &keys,
                   const std::vector<std::vector<float>> &values) override;

  int FakePutParameter(base::ConstArray<uint64_t> keys, float *values) override;

 private:
  std::vector<int> GetServerThreadIDs();
  int SelectServerThreadID() const;
  void Init();
  DSM *dsm_;

  uint64_t rpcIDAcc_ = 0;

  std::vector<int> serverThreadIdsRoutedTo_;
  std::unordered_map<uint64_t, int *> rpcId2PollMap_;
};

FACTORY_REGISTER(BaseParameterClient, PetPSClient,
                 PetPSClient, const std::string &, int, int);
}  // namespace petps