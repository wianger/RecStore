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

int64_t numel(const at::IntArrayRef shape) {
  int64_t ret = 1;
  for (auto i : shape) {
    ret *= i;
  }
  return ret;
}

enum IPCTensorType {
  kCPUIPCTensor = 0,
  kGPUIPCTensor = 1,
};

class IPCTensorMemoryHandle {
  static constexpr int kShapeDimMax = 3;

public:
  IPCTensorMemoryHandle(const std::string &name, at::IntArrayRef shape,
                        at::ScalarType dtype)
      : dtype_(dtype) {
    type_ = kCPUIPCTensor;
    memset(shape_, -1, kShapeDimMax * sizeof(int));
    for (int i = 0; i < shape.size(); i++) {
      shape_[i] = shape[i];
    }
    shape_dim_ = shape.size();
    assert(shape_dim_ <= kShapeDimMax);
    assert(name.size() < 96);
    strcpy(name_, name.c_str());
    size_ = numel(shape_vec()) * c10::elementSize(dtype_);
  }

  IPCTensorMemoryHandle(const std::string &name, at::IntArrayRef shape,
                        at::ScalarType dtype, int64_t dev_id,
                        cudaIpcMemHandle_t memHandle, void *dev_ptr)
      : dev_ptr_(dev_ptr), dev_id_(dev_id), dtype_(dtype),
        memHandle_(memHandle) {
    type_ = kGPUIPCTensor;
    memset(shape_, -1, kShapeDimMax * sizeof(int));
    for (int i = 0; i < shape.size(); i++) {
      shape_[i] = shape[i];
    }
    shape_dim_ = shape.size();
    assert(shape_dim_ <= kShapeDimMax);
    assert(name.size() < 96);
    strcpy(name_, name.c_str());
    size_ = numel(shape_vec()) * c10::elementSize(dtype_);

    std::cout << "this = " << this << std::endl;
    std::cout << "this->dev_ptr = " << this->dev_ptr_ << std::endl;
    std::cout << "this->type_= " << this->type_ << std::endl;
  }

  std::vector<int64_t> shape_vec() const {
    std::vector<int64_t> ret;
    for (int i = 0; i < shape_dim_; i++) {
      ret.push_back(shape_[i]);
    }
    return ret;
  }

  int64_t WholeSizeInCPU() const {
    if (type_ == kCPUIPCTensor)
      return sizeof(IPCTensorMemoryHandle) + size_;
    else
      return sizeof(IPCTensorMemoryHandle);
  }

  static int64_t WholeSizeInCPU(IPCTensorType type, at::IntArrayRef shape,
                                at::ScalarType dtype) {
    if (type == kCPUIPCTensor)
      return sizeof(IPCTensorMemoryHandle) +
             numel(shape) * c10::elementSize(dtype);
    else
      return sizeof(IPCTensorMemoryHandle);
  }

  void *GetTensorPtr() const {
    if (type_ == kCPUIPCTensor) {
      return (void *)host_ptr_;
    } else {
      return dev_ptr_;
    }
  }

  auto CheckMagic() const { assert(magic_ == 0xdeadbeef); }

  auto GetDtype() const { return dtype_; }

  auto GetName() const { return name_; }

  auto GetDeviceID() const {
    assert(type_ == kGPUIPCTensor);
    return dev_id_;
  }

  auto GetIPCType() const { return type_; }

  cudaIpcMemHandle_t GetCUDAIPCMemHandle() const {
    assert(type_ == kGPUIPCTensor);
    return memHandle_;
  }

private:
  IPCTensorType type_;
  char name_[96];
  int shape_[kShapeDimMax];
  int shape_dim_;
  int64_t size_;
  at::ScalarType dtype_;
  const int magic_ = 0xdeadbeef;

  union {
    struct {
      void *dev_ptr_;
      cudaIpcMemHandle_t memHandle_;
      int dev_id_;
    };
    struct {
      char host_ptr_[0];
    };
  };
};

class IPCMemory {
  static constexpr int kMaxRegTensorNum = 10;
  static constexpr int64_t kShmSize = 2 * (1024 * 1024 * 1024LL);

  struct IPCShmRegion {
    std::atomic<int64_t> accumulated_offset_;
    int64_t malloced_offsets_[kMaxRegTensorNum];
    IPCTensorMemoryHandle handles[0];
  };

  IPCMemory()
      : mapping_("/dev/shm/recstore_ipc_memory", 0, kShmSize,
                 folly::MemoryMapping::writable().setPrefault(true)),
        header_((IPCShmRegion *)mapping_.writableRange().begin()) {}

public:
  static IPCMemory *GetInstance() {
    static IPCMemory instance;
    return &instance;
  }

  IPCTensorMemoryHandle *RegisterMemory(std::string name, at::IntArrayRef shape,
                                        at::ScalarType dtype) {
    int64_t size = numel(shape) * c10::elementSize(dtype);
    int64_t obj_size_in_shm = IPCTensorMemoryHandle::WholeSizeInCPU(
        IPCTensorType::kCPUIPCTensor, shape, dtype);

    // FAA the atomic variable in the header
    int64_t offset = header_->accumulated_offset_.fetch_add(obj_size_in_shm);
    assert(offset + obj_size_in_shm < kShmSize);

    auto *p = new ((char *)header_ + offset)
        IPCTensorMemoryHandle(name, shape, dtype);

    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t &malloced_offset = header_->malloced_offsets_[i];
      if (malloced_offset == 0) {
        malloced_offset = offset;
        break;
      }
    }
    p->CheckMagic();
    return p;
  }

  IPCTensorMemoryHandle *RegisterGPUMemory(std::string name,
                                           at::IntArrayRef shape,
                                           at::ScalarType dtype, int dev_id) {
    int64_t size = numel(shape) * c10::elementSize(dtype);
    nv::CudaDeviceRestorer _;
    cudaSetDevice(dev_id);

    void *d_ptr;
    cudaMalloc(&d_ptr, size);
    std::cout << "xmh1 = " << d_ptr << std::endl;
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, d_ptr);

    // printf("Put memhandle = %s\n", memHandle.reserved);

    int64_t obj_size_in_shm = IPCTensorMemoryHandle::WholeSizeInCPU(
        IPCTensorType::kGPUIPCTensor, shape, dtype);

    // FAA the atomic variable in the header
    int64_t offset = header_->accumulated_offset_.fetch_add(obj_size_in_shm);
    assert(offset + obj_size_in_shm < kShmSize);

    auto *p = new ((char *)header_ + offset)
        IPCTensorMemoryHandle(name, shape, dtype, dev_id, memHandle, d_ptr);

    std::cout << "new p = " << p << std::endl;
    std::cout << "p->GetIPCType=" << p->GetIPCType() << std::endl;
    std::cout << "p->GetTensorPtr = " << p->GetTensorPtr() << std::endl;

    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t &malloced_offset = header_->malloced_offsets_[i];
      if (malloced_offset == 0) {
        malloced_offset = offset;
        break;
      }
    }
    p->CheckMagic();
    return p;
  }

  IPCTensorMemoryHandle *GetHandle(const std::string &name) {
    for (int i = 0; i < kMaxRegTensorNum; i++) {
      int64_t offset = header_->malloced_offsets_[i];
      if (offset == 0) {
        break;
      }
      auto *p = (IPCTensorMemoryHandle *)((char *)header_ + offset);
      p->CheckMagic();
      if (strcmp(p->GetName(), name.c_str()) == 0) {
        return p;
      }
    }
    std::cerr << "not find handle for name = " << name << std::endl;
    assert(false);
  }

  void ClearIPCMemory() {
    for (int i = 0; i < kMaxRegTensorNum; i++) {
      header_->malloced_offsets_[i] = 0;
    }
    header_->accumulated_offset_ = sizeof(IPCShmRegion);
  }

private:
  folly::MemoryMapping mapping_;
  IPCShmRegion *header_;
};

class IPCTensorFactory : public torch::CustomClassHolder {
public:
  static void ClearIPCMemory() { IPCMemory::GetInstance()->ClearIPCMemory(); }

  static torch::Tensor NewIPCTensor(const std::string &name,
                                    const at::IntArrayRef shape,
                                    const at::ScalarType dtype) {
    int64_t size_in_bytes = numel(shape) * c10::elementSize(dtype);
    auto handle = IPCMemory::GetInstance()->RegisterMemory(name, shape, dtype);
    assert(handle->GetIPCType() == kCPUIPCTensor);
    auto tensor = torch::from_blob(
        handle->GetTensorPtr(), handle->shape_vec(),
        torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
    return tensor;
  }

  static torch::Tensor NewIPCGPUTensor(const std::string &name,
                                       const at::IntArrayRef shape,
                                       const at::ScalarType dtype,
                                       const int64_t dev_id) {
    int64_t size_in_bytes = numel(shape) * c10::elementSize(dtype);
    auto handle =
        IPCMemory::GetInstance()->RegisterGPUMemory(name, shape, dtype, dev_id);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    assert(handle->GetIPCType() == kGPUIPCTensor);

    std::cout << "handle = " << handle << std::endl;
    std::cout << "handle->GetTensorPtr() = " << handle->GetTensorPtr()
              << std::endl;

    auto tensor = torch::from_blob(handle->GetTensorPtr(), handle->shape_vec(),
                                   torch::TensorOptions()
                                       .dtype(handle->GetDtype())
                                       .device(torch::kCUDA, dev_id));
    return tensor;
  }

  static torch::Tensor GetIPCTensorFromName(const std::string &name) {
    IPCTensorMemoryHandle *handle = IPCMemory::GetInstance()->GetHandle(name);
    handle->CheckMagic();
    if (handle->GetIPCType() == kGPUIPCTensor) {
      void *ptr;
      cudaIpcOpenMemHandle(&ptr, handle->GetCUDAIPCMemHandle(),
                           cudaIpcMemLazyEnablePeerAccess);
      // printf("Get memhandle = %s\n", handle->memHandle_.reserved);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
      return torch::from_blob(ptr, handle->shape_vec(),
                              torch::TensorOptions()
                                  .dtype(handle->GetDtype())
                                  .device(torch::kCUDA, handle->GetDeviceID()));

    } else {
      void *ptr = (void *)handle->GetTensorPtr();
      return torch::from_blob(
          ptr, handle->shape_vec(),
          torch::TensorOptions().dtype(handle->GetDtype()).device(torch::kCPU));
    }
  }

private:
};

void RegisterIPCTensorFactory(torch::Library &m) {
  m.class_<IPCTensorFactory>("IPCTensorFactory")
      .def_static("ClearIPCMemory", &IPCTensorFactory::ClearIPCMemory)
      .def_static("NewIPCTensor", &IPCTensorFactory::NewIPCTensor)
      .def_static("NewIPCGPUTensor", &IPCTensorFactory::NewIPCGPUTensor)
      .def_static("GetIPCTensorFromName",
                  &IPCTensorFactory::GetIPCTensorFromName);
}

} // namespace recstore