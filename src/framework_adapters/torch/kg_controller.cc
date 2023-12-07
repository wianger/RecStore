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
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "torch_utils.h"

namespace recstore {

class KGCacheController : public torch::CustomClassHolder {
 public:
  KGCacheController(int64_t num_gpus, int64_t L) {
    // num_gpus_ = torch::cuda::device_count();
    num_gpus_ = num_gpus;
    L_ = L;
  }

  void RegTensorsPerProcess() {
    // IPCTensorFactory::ListIPCTensors();
    for (int rank = 0; rank < num_gpus_; ++rank) {
      nv::CudaDeviceRestorer _;
      CUDA_CHECK(cudaSetDevice(rank));
      input_keys_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("input_keys_{}", rank)));
      input_keys_neg_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("input_keys_neg_{}", rank)));

      // cache tensor
      cache_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("embedding_cache_{}", rank)));

      // L buffer of input ids
      cached_id_circle_buffer_.push_back(std::vector<torch::Tensor>());
      for (int j = 0; j < L_; ++j) {
        cached_id_circle_buffer_[rank].push_back(
            IPCTensorFactory::GetIPCTensorFromName(
                folly::sformat("cached_sampler_r{}_{}", rank, j)));
      }
      // step tensor
      step_tensor_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("step_r{}", rank)));
    }
  }

  void ProcessOneStep() {
    // for (int i = 0; i < num_gpus_; ++i) {
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
    // }
  }

 private:
  int num_gpus_;
  int L_;
  std::vector<torch::Tensor> input_keys_per_rank_;
  std::vector<torch::Tensor> input_keys_neg_per_rank_;
  std::vector<torch::Tensor> cache_per_rank_;
  std::vector<torch::Tensor> step_tensor_per_rank_;
  std::vector<std::vector<torch::Tensor>> cached_id_circle_buffer_;
};

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def(torch::init<int64_t, int64_t>())
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess)
      .def("ProcessOneStep", &KGCacheController::ProcessOneStep);
}

}  // namespace recstore
