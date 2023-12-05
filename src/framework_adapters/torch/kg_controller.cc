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
  KGCacheController(int64_t num_gpus) {
    // num_gpus_ = torch::cuda::device_count();
    num_gpus_ = num_gpus;
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
    }
  }

 private:
  void ProcessOneStep() { ; }

  int num_gpus_;
  std::vector<torch::Tensor> input_keys_per_rank_;
  std::vector<torch::Tensor> cache_per_rank_;
};

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def(torch::init<int64_t>())
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess);
}

}  // namespace recstore
