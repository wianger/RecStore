#include <ATen/cuda/CUDAContext.h>
#include <folly/Format.h>
#include <folly/system/MemoryMapping.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "IPCTensor.h"
#include "base/base.h"
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/json.h"
#include "torch_utils.h"

namespace recstore {

class KGCacheController : public torch::CustomClassHolder {
 public:
  KGCacheController(const std::string &json_str,
                    const std::vector<std::vector<int64_t>> &cached_range) {
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config["num_gpus"];
    L_ = json_config["L"];
    kForwardItersPerStep_ = json_config["kForwardItersPerStep"];
    clr_ = json_config["clr"];

    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);
  }

  void RegTensorsPerProcess() {
    // IPCTensorFactory::ListIPCTensors();
    for (int rank = 0; rank < num_gpus_; ++rank) {
      nv::CudaDeviceRestorer _;
      CUDA_CHECK(cudaSetDevice(rank));
      input_keys_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_{}", rank)));
      input_keys_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_neg_{}", rank)));

      // cache tensor
      cache_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("embedding_cache_{}", rank)));

      // L buffer of input ids
      cached_id_circle_buffer_.push_back(std::vector<torch::Tensor>());
      for (int j = 0; j < L_; ++j) {
        cached_id_circle_buffer_[rank].push_back(
            IPCTensorFactory::FindIPCTensorFromName(
                folly::sformat("cached_sampler_r{}_{}", rank, j)));
      }
      // step tensor
      step_tensor_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("step_r{}", rank)));

      backward_grads_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_{}", rank)));

      backward_grads_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_neg_{}", rank)));
    }

    full_emb_ = IPCTensorFactory::FindIPCTensorFromName("full_emb");
  }

  void ProcessOneStep() {
    torch::AutoGradMode guard_false(false);
    // show tensors of the first rank
    std::cout << "input_keys_per_rank" << std::endl;
    std::cout << toString(input_keys_per_rank_[0]) << std::endl;
    std::cout << "input_keys_neg_per_rank_" << std::endl;
    std::cout << toString(input_keys_neg_per_rank_[0]) << std::endl;
    static int cnt = 0;
    std::cout << "cached_id_circle_buffer" << std::endl;
    std::cout << "Step " << step_tensor_per_rank_[0][cnt].item<int64_t>()
              << " ";
    std::cout << toString(cached_id_circle_buffer_[0][cnt]) << std::endl;
    cnt = (cnt + 1) % cached_id_circle_buffer_[0].size();

    auto input_keys_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(input_keys_per_rank_);
    auto backward_grads_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(backward_grads_per_rank_);

    ProcessBackwardSync(input_keys_per_rank_tensors,
                        backward_grads_per_rank_tensors);
    if (kForwardItersPerStep_ > 1) {
      auto input_keys_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(input_keys_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_per_rank_);
      ProcessBackwardSync(input_keys_neg_per_rank_tensors,
                          backward_grads_neg_per_rank_tensors);
    }
  }

  std::vector<torch::Tensor> split_keys_to_shards(
      const torch::Tensor keys,
      std::vector<std::vector<int64_t>> cached_range) {
    std::vector<torch::Tensor> in_each_rank_cache_mask;
    CHECK_EQ(cached_range.size(), num_gpus_);

    for (int shard_no = 0; shard_no < num_gpus_; shard_no++) {
      int64_t start = cached_range[shard_no][0];
      int64_t end = cached_range[shard_no][1];

      torch::Tensor in_this_rank =
          keys.greater_equal(start).logical_and(keys.less(end));
      in_each_rank_cache_mask.push_back(in_this_rank);
    }
    return in_each_rank_cache_mask;
  }

  //  cached_range:
  //        [ (start, end ),  # rank0
  //         (start, end),    # rank1
  //         (start, end),  ....
  //         (start, end),
  //         (start, end),    # rank7
  //        ]
  void ProcessBackwardSync(const std::vector<torch::Tensor> &input_keys,
                           const std::vector<torch::Tensor> &input_grads

  ) {
    // auto input_keys = input_keys_per_rank_;
    // auto input_grads = backward_grads_per_rank_;

    std::vector<std::vector<torch::Tensor>> shuffled_keys;
    std::vector<std::vector<torch::Tensor>> shuffled_grads;
    shuffled_keys.resize(num_gpus_);
    shuffled_grads.resize(num_gpus_);

    for (int rank = 0; rank < num_gpus_; rank++) {
      // CHECK(!input_keys[rank].is_cuda());
      auto in_each_rank_cache_mask =
          split_keys_to_shards(input_keys[rank], cached_range_);

      std::vector<torch::Tensor> sharded_keys_in_this_rank =
          TensorUtil::IndexVectors(input_keys[rank], in_each_rank_cache_mask);
      std::vector<torch::Tensor> sharded_grads_in_this_rank =
          TensorUtil::IndexVectors(input_grads[rank], in_each_rank_cache_mask);

      // shuffle keys and grads
      for (int i = 0; i < num_gpus_; i++) {
        shuffled_keys[i].push_back(sharded_keys_in_this_rank[i]);
        shuffled_grads[i].push_back(sharded_grads_in_this_rank[i]);
      }
    }
    // shuffle done

    SGDGradUpdate(shuffled_keys, shuffled_grads);
  }

  void SGDGradUpdate(
      const std::vector<std::vector<torch::Tensor>> &shuffled_keys,
      const std::vector<std::vector<torch::Tensor>> &shuffled_grads) {
    // std::vector<std::vector<torch::Tensor>> shuffled_keys_cuda;
    // std::vector<std::vector<torch::Tensor>> shuffled_grads_cuda;

    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // update per rank cache
        // CUDA_CHECK(cudaSetDevice(rank));
        torch::Device device(torch::kCUDA, rank);
        auto in_rank_keys =
            shuffled_keys[rank][j].to(device, true) - cached_range_[rank][0];
        auto shuffle_grads_cuda = shuffled_grads[rank][j].to(device, true);
        cache_per_rank_[rank].index_add_(0, in_rank_keys, -shuffle_grads_cuda);
      }
    }

    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // LOG(WARNING) << rank << ": shuffled_keys[rank][j] shape="
        //              << shuffled_keys[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_keys[rank][j]"
                     << toString(shuffled_keys[rank][j], false);
        // LOG(WARNING) << rank << ": shuffled_grads[rank][j] shape="
        //              << shuffled_grads[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_grads[rank][j]"
                     << toString(shuffled_grads[rank][j], false);

        // update full emb
        full_emb_.index_add_(0, shuffled_keys[rank][j].cpu(),
                             -shuffled_grads[rank][j].cpu());
      }
    }

    // full_emb_.zero_();
    // LOG(ERROR) << "is_cpu" << full_emb_.is_cpu();
    // LOG(ERROR) << "full_emb.addr=" << full_emb_.data_ptr<float>();
    // LOG(ERROR) << toString(full_emb_, false);
  }

  void ProcessBackwardAsync() {}

 private:
  // config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;

  // state tensor
  torch::Tensor full_emb_;
  std::vector<torch::Tensor> cache_per_rank_;

  // runtime tensor
  std::vector<torch::Tensor> step_tensor_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_per_rank_;

  std::vector<std::vector<torch::Tensor>> cached_id_circle_buffer_;
};

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def(torch::init<const std::string,
                       const std::vector<std::vector<int64_t>>>())
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess)
      .def("ProcessOneStep", &KGCacheController::ProcessOneStep);
}

}  // namespace recstore
