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
    for (int i = 0; i < num_gpus_; ++i) {
      nv::CudaDeviceRestorer _;
      CUDA_CHECK(cudaSetDevice(i));
      input_keys_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("input_keys_{}", i)));
      cache_per_rank_.push_back(IPCTensorFactory::GetIPCTensorFromName(
          folly::sformat("embedding_cache_{}", i)));

      cached_id_circle_buffer_.push_back(std::vector<torch::Tensor>());
      for (int j = 0; j < L_; ++j) {
        cached_id_circle_buffer_[i].push_back(
            IPCTensorFactory::GetIPCTensorFromName(
                folly::sformat("cached_sampler_{}_{}", i, j)));
      }
    }
  }

  void ProcessOneStep() {
    for (int i = 0; i < num_gpus_; ++i) {
      std::cout << input_keys_per_rank_[i];
    }
  }

 private:
  int num_gpus_;
  int L_;
  std::vector<torch::Tensor> input_keys_per_rank_;
  std::vector<torch::Tensor> cache_per_rank_;
  std::vector<std::vector<torch::Tensor>> cached_id_circle_buffer_;
};

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def(torch::init<int64_t, int64_t>())
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess)
      .def("ProcessOneStep", &KGCacheController::ProcessOneStep);
}

}  // namespace recstore
