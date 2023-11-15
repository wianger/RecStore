#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "base/cu_utils.cuh"

#include "IPCGPUMemoryHandle.pb.h"

namespace recstore {
class IPCGPUMemoryHandle : public torch::CustomClassHolder {
public:
  IPCGPUMemoryHandle() {}
  IPCGPUMemoryHandle(void *ptr, int64_t dev_id, at::IntArrayRef shape,
                     at::ScalarType dtype, cudaIpcMemHandle_t memHandle)
      : ptr(ptr), dev_id(dev_id), shape(shape), dtype(dtype),
        memHandle(memHandle) {}

  std::string __repr__() const {
    IPCGPUMemoryHandlePB object;
    object.set_ptr((int64_t)ptr);
    object.set_dev_id(dev_id);
    for (auto i : shape) {
      object.add_shape(i);
    }
    object.set_dtype((int)dtype);
    object.set_mem_handle(std::string(memHandle.reserved, 64));

    std::string serializedData;
    bool ret;
    ret = object.SerializeToString(&serializedData);

    assert(ret);
    return serializedData;
  }

  static c10::intrusive_ptr<IPCGPUMemoryHandle> CreateFromString(const std::string &str) {
    IPCGPUMemoryHandlePB object;
    object.ParseFromString(str);
    IPCGPUMemoryHandle ret;
    ret.ptr = (void *)object.ptr();
    ret.dev_id = object.dev_id();

    std::vector<int64_t> shape(object.shape().begin(), object.shape().end());
    ret.shape = at::IntArrayRef(shape);
    ret.dtype = (at::ScalarType)object.dtype();

    memcpy(&ret.memHandle, object.mem_handle().c_str(), sizeof(ret.memHandle));
    return c10::make_intrusive<IPCGPUMemoryHandle>(ret);
  }

  void *ptr;
  int dev_id;
  at::IntArrayRef shape;
  at::ScalarType dtype;
  cudaIpcMemHandle_t memHandle;
};

class IPCGPUMemory {
  IPCGPUMemory() = default;

public:
  static IPCGPUMemory *GetInstance() {
    static IPCGPUMemory instance;
    return &instance;
  }

  c10::intrusive_ptr<IPCGPUMemoryHandle> RegisterMemory(std::string name,
                                                        int dev_id, int size,
                                                        at::IntArrayRef shape,
                                                        at::ScalarType dtype) {
    nv::CudaDeviceRestorer _;
    cudaSetDevice(dev_id);
    void *d_ptr;
    cudaMalloc(&d_ptr, size);
    cudaIpcMemHandle_t memHandle;
    cudaIpcGetMemHandle(&memHandle, d_ptr);

    printf("Put memhandle = %s\n", memHandle.reserved);

    memory_map_[name] = c10::make_intrusive<IPCGPUMemoryHandle>(
        d_ptr, dev_id, shape, dtype, memHandle);
    return memory_map_[name];
  }

private:
  std::unordered_map<std::string, c10::intrusive_ptr<IPCGPUMemoryHandle>>
      memory_map_;
};

class IPCTensorFactory : public torch::CustomClassHolder {
public:
  static c10::intrusive_ptr<IPCGPUMemoryHandle>
  NewIPCGPUTensor(const int64_t dev_id, const at::IntArrayRef shape,
                  const std::string &name, const at::ScalarType dtype) {
    int numel = 1;
    for (auto i : shape) {
      numel *= i;
    }
    int size_in_bytes = numel * c10::elementSize(dtype);

    auto handle = IPCGPUMemory::GetInstance()->RegisterMemory(
        name, dev_id, size_in_bytes, shape, dtype);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // std::cout << "xmh" << std::endl;
    // void *get_ptr;
    // cudaIpcOpenMemHandle(&get_ptr, handle->memHandle,
    //                      cudaIpcMemLazyEnablePeerAccess);
    // C10_CUDA_KERNEL_LAUNCH_CHECK();
    // std::cout << "xmh done" << std::endl;
    return handle;
  }

  static torch::Tensor GetIPCGPUTensorFromHandleLocal(
      c10::intrusive_ptr<IPCGPUMemoryHandle> handle) {
    auto tensor = torch::from_blob(handle->ptr, handle->shape,
                                   torch::TensorOptions()
                                       .dtype(handle->dtype)
                                       .device(torch::kCUDA, handle->dev_id));
    return tensor;
  }

  static torch::Tensor
  GetIPCGPUTensorFromHandle(c10::intrusive_ptr<IPCGPUMemoryHandle> handle) {
    void *ptr;
    printf("Get memhandle = %s\n", handle->memHandle.reserved);
    cudaIpcOpenMemHandle(&ptr, handle->memHandle,
                         cudaIpcMemLazyEnablePeerAccess);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return torch::from_blob(handle->ptr, handle->shape,
                            torch::TensorOptions()
                                .dtype(handle->dtype)
                                .device(torch::kCUDA, handle->dev_id));
  }

private:
};

void RegisterIPCTensorFactory(torch::Library &m) {
  m.class_<IPCGPUMemoryHandle>("IPCGPUMemoryHandle")
      .def("__repr__", &IPCGPUMemoryHandle::__repr__)
      .def_static("CreateFromString", &IPCGPUMemoryHandle::CreateFromString);

  // .def(torch::init<>())
  //     .def(torch::init<void *, int64_t, at::IntArrayRef, at::ScalarType,
  //                      cudaIpcMemHandle_t>());

  // .def_readwrite("ptr", &IPCGPUMemoryHandle::ptr)
  // .def_readwrite("dev_id", &IPCGPUMemoryHandle::dev_id)
  // .def_readwrite("shape", &IPCGPUMemoryHandle::shape)
  // .def_readwrite("dtype", &IPCGPUMemoryHandle::dtype)
  // .def_readwrite("memHandle", &IPCGPUMemoryHandle::memHandle);

  m.class_<IPCTensorFactory>("IPCTensorFactory")
      .def_static("NewIPCGPUTensor", &IPCTensorFactory::NewIPCGPUTensor)
      .def_static("GetIPCGPUTensorFromHandleLocal",
                  &IPCTensorFactory::GetIPCGPUTensorFromHandleLocal)
      .def_static("GetIPCGPUTensorFromHandle",
                  &IPCTensorFactory::GetIPCGPUTensorFromHandle);
}

} // namespace recstore