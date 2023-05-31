#pragma once

#include <cstdint>
#include <future>
#include <string>
#include <vector>

#include "flatc.h"
#include "parameters.h"
#include "ps.grpc.pb.h"
#include "ps.pb.h"
#include "base/array.h"

#include "folly/Portability.h"
#include "folly/executors/CPUThreadPoolExecutor.h"
#include "folly/init/Init.h"

using grpc::Channel;
using grpc::ClientContext;
using grpc::Status;
using xmhps::CommandRequest;
using xmhps::CommandResponse;
using xmhps::GetParameterRequest;
using xmhps::GetParameterResponse;
using xmhps::PSCommand;
using xmhps::PutParameterRequest;
using xmhps::PutParameterResponse;

class ParameterClient {
 public:
  explicit ParameterClient(const std::string &host, int port, int shard);
  ~ParameterClient() {}

  bool GetParameter(ConstArray<uint64_t> &keys, std::vector<std::vector<float>> *values,
                    bool perf = true);
  bool GetParameter(ConstArray<unsigned int> &keys,
                    std::vector<std::vector<float>> *values, bool perf = true);

	// this interface assume all keys with the same embedding dimension
  bool GetParameter(ConstArray<uint64_t> &keys, float *values, bool perf = true);

  inline int shard() const {
    return shard_;
  }

  bool ClearPS();

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path);

  bool PutParameter(const std::vector<uint64_t> &keys,
                    const std::vector<std::vector<float>> &values);

 private:
  bool Initialize() {
    return true;
  }

  std::string host_;
  int port_;
  int shard_;
  int nr_clients_;
  std::vector<std::future<bool>> futures_;
  std::vector<float> cache_;
  std::vector<int32_t> offset_;
  std::unique_ptr<folly::CPUThreadPoolExecutor> future_pool_;
  std::vector<int> get_param_key_sizes_;
  std::vector<GetParameterRequest> get_param_requests_;
  std::vector<GetParameterResponse> get_param_responses_;
  std::shared_ptr<Channel> channel_;
  std::vector<std::unique_ptr<xmhps::ParameterService::Stub>> stubs_;
};

// class MultiPSClient {
//   int GetShard(uint64_t key, int nr_shards) const { return key % nr_shards;
//   }

//   bool GetParameter(const std::vector<uint64_t>& keys,
//                     std::vector<std::vector<const float>>* values) {
//     if (keys.empty()) {
//       return true;
//     }
//     values->clear();
//     for (int i = 0; i < clients_.size(); ++i) shard_keys_[i].clear();
//     for (auto key : keys) {
//       auto shard = GetShard(key, clients_.size());
//       shard_keys_[shard].push_back(key);
//     }

//     for (int i = 0; i < clients_.size(); ++i) {
//       if (shard_keys_[i].empty()) continue;
//       get_batch_holder_[i] = batch_client_->GetParameter(i,
//       shard_keys_[i]);
//     }
//     bool res = true;
//     for (int i = 0; i < clients_.size(); ++i) {
//       if (shard_keys_[i].empty() || !get_batch_holder_[i]) {
//         continue;
//       }
//       while (!get_batch_holder_[i]->done) {
//         base::SleepForMilliseconds(1);
//       }
//       if (!get_batch_holder_[i]->parameters) {
//         LOG(INFO) << "shard " << i << " parameters error";
//         res = false;
//       }
//     }
//     if (!res) return false;
//     for (auto key : keys) {
//       int shard = GetShard(key, clients_.size());
//       values->push_back(get_batch_holder_[shard]->GetSingleParameter(key));
//     }
//     return true;
//   }

//   std::vector<std::unique_ptr<ParameterClient>> clients_;
//   std::vector<> get_batch_holder_;
// };