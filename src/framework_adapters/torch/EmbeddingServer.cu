#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <string>
#include <unordered_map>

#include "cu_utils.cuh"

class IPCGPUMemory {
  struct IPCGPUMemoryHandle {
    void *ptr;
    int dev_id;
    cudaIpcMemHandle_t memHandle;
  };
  IPCGPUMemory() = default;

public:
  static IPCGPUMemory *GetInstance() {
    static IPCGPUMemory instance;
    return &instance;
  }

  void *RegisterMemory(std::string name, int dev_id, int size) {
    CudaDeviceRestorer _;
    cudaSetDevice(dev_id);
    void *ptr;
    cudaMalloc(&ptr, size);
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, ptr);
    memory_map_[name] = IPCGPUMemoryHandle{ptr, dev_id, memHandle};
    return ptr;
  }

private:
  std::unordered_map<std::string, IPCGPUMemoryHandle> memory_map_;
};

class EmbeddingServer : public torch::CustomClassHolder {
public:
  torch::Tensor NewGPUTensor(int32_t dev_id, at::IntArrayRef shape,
                             const std::string &name, at::ScalarType dtype) {
    int numel = 1;
    for (auto i : shape) {
      numel *= i;
    }
    int size_in_bytes = numel * dtype.itemsize();

    void *ptr = IPCGPUMemory::GetInstance()->RegisterMemory(name, dev_id,
                                                            size_in_bytes);

    return torch::from_blob(
        ptr, shape,
        torch::TensorOptions().dtype(dtype).device(torch::kCUDA, dev_id));
  }

private:
};