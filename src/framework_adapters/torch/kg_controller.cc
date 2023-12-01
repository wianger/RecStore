#include <ATen/cuda/CUDAContext.h>
#include <folly/system/MemoryMapping.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "base/cu_utils.cuh"
#include "base/debug_utils.h"

namespace recstore {


// class IPCGPUMemoryHandle {
//   static constexpr int kShapeDimMax = 3;

//  public:
//   IPCGPUMemoryHandle() {}
//   IPCGPUMemoryHandle(const std::string &name, void *ptr, int64_t dev_id,
//                      at::IntArrayRef shape, at::ScalarType dtype,
//                      cudaIpcMemHandle_t memHandle)
//       : ptr_(ptr), dev_id_(dev_id), dtype_(dtype), memHandle_(memHandle) {
//     memset(shape_, -1, kShapeDimMax * sizeof(int));
//     for (int i = 0; i < shape.size(); i++) {
//       shape_[i] = shape[i];
//     }
//     shape_dim_ = shape.size();
//     assert(shape_dim_ <= kShapeDimMax);
//     strcpy(name_, name.c_str());
//   }

//   std::vector<int64_t> shape_vec() const {
//     std::vector<int64_t> ret;
//     for (int i = 0; i < shape_dim_; i++) {
//       ret.push_back(shape_[i]);
//     }
//     return ret;
//   }

//   // private:
//   char name_[96];
//   void *ptr_;
//   int dev_id_;
//   int shape_[kShapeDimMax];
//   int shape_dim_;
//   at::ScalarType dtype_;
//   cudaIpcMemHandle_t memHandle_;
//   const int magic_ = 0xdeadbeef;
// };

// class IPCGPUMemory {
//   static constexpr int kMaxRegTensorNum = 10;

//   IPCGPUMemory()
//       : mapping_("/tmp/recstore_ipc_gpu_memory", 0,
//                  sizeof(IPCGPUMemoryHandle) * kMaxRegTensorNum,
//                  folly::MemoryMapping::writable().setPrefault(true)),
//         start_((IPCGPUMemoryHandle *)mapping_.writableRange().begin()) {}

//  public:
//   static IPCGPUMemory *GetInstance() {
//     static IPCGPUMemory instance;
//     return &instance;
//   }

//   IPCGPUMemoryHandle *RegisterMemory(std::string name, int dev_id, int64_t size,
//                                      at::IntArrayRef shape,
//                                      at::ScalarType dtype) {
//     nv::CudaDeviceRestorer _;
//     cudaSetDevice(dev_id);

//     void *d_ptr;
//     cudaMalloc(&d_ptr, size);
//     cudaIpcMemHandle_t memHandle;
//     cudaIpcGetMemHandle(&memHandle, d_ptr);

//     printf("Put memhandle = %s\n", memHandle.reserved);

//     static int regTensorNum_ = 0;
//     auto p = start_ + regTensorNum_;
//     new (p) IPCGPUMemoryHandle(name, d_ptr, dev_id, shape, dtype, memHandle);
//     regTensorNum_ += 1;
//     return p;
//   }

//   IPCGPUMemoryHandle *GetHandle(const std::string &name) {
//     auto *p = start_;
//     for (int i = 0; i < kMaxRegTensorNum; i++) {
//       assert(p->magic_ == 0xdeadbeef);
//       if (strcmp(p->name_, name.c_str()) == 0) {
//         return p;
//       }
//       p++;
//     }
//     assert(false);
//   }

//  private:
//   folly::MemoryMapping mapping_;
//   IPCGPUMemoryHandle *start_;
// };

// class IPCTensorFactory : public torch::CustomClassHolder {
//  public:
//   static torch::Tensor NewIPCGPUTensor(const int64_t dev_id,
//                                        const at::IntArrayRef shape,
//                                        const std::string &name,
//                                        const at::ScalarType dtype) {
//     int numel = 1;
//     for (auto i : shape) {
//       numel *= i;
//     }
//     int64_t size_in_bytes = numel * c10::elementSize(dtype);

//     auto handle = IPCGPUMemory::GetInstance()->RegisterMemory(
//         name, dev_id, size_in_bytes, shape, dtype);
//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     auto tensor = torch::from_blob(handle->ptr_, handle->shape_vec(),
//                                    torch::TensorOptions()
//                                        .dtype(handle->dtype_)
//                                        .device(torch::kCUDA, handle->dev_id_));
//     return tensor;
//   }

//   static torch::Tensor GetIPCGPUTensorFromName(const std::string &name) {
//     IPCGPUMemoryHandle *handle = IPCGPUMemory::GetInstance()->GetHandle(name);

//     void *ptr;
//     TORCH_CHECK(handle->magic_ == 0xdeadbeef, "magic check error");
//     printf("Get memhandle = %s\n", handle->memHandle_.reserved);
//     cudaIpcOpenMemHandle(&ptr, handle->memHandle_,
//                          cudaIpcMemLazyEnablePeerAccess);

//     C10_CUDA_KERNEL_LAUNCH_CHECK();
//     return torch::from_blob(ptr, handle->shape_vec(),
//                             torch::TensorOptions()
//                                 .dtype(handle->dtype_)
//                                 .device(torch::kCUDA, handle->dev_id_));
//   }

//  private:
// };


class KGCacheController {
 public:
  KGCacheController() {}

  void Start() { ; }

  torch::Tensor NewShmTensor(const std::string &name, at::IntArrayRef shape,
                             at::ScalarType dtype, int64_t dev_id) {}

 private:
  void ProcessOneStep() { ; }

  int num_gpus_;
  std::vector<torch::Tensor *> input_keys_per_rank_;
  std::vector<torch::Tensor *> cache_per_rank_;
};
}  // namespace recstore

asdasd