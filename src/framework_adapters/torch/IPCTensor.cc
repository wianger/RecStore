#include <ATen/cuda/CUDAContext.h>
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

int64_t numel(const at::IntArrayRef shape) {
  int64_t ret = 1;
  for (auto i : shape) {
    ret *= i;
  }
  return ret;
}

c10::intrusive_ptr<SlicedTensor> IPCTensorFactory::GetSlicedIPCTensorFromName(
    const std::string &name) {
  IPCTensorMemoryHandle *handle = IPCMemory::GetInstance()->GetHandle(name);
  if (nullptr == handle) {
    LOG(FATAL) << "IPCTensor " << name << " not found";
  }
  return c10::make_intrusive<SlicedTensor>(handle);
}

void RegisterIPCTensorFactory(torch::Library &m) {
  m.class_<SlicedTensor>("SlicedTensor")
      .def("GetSlicedTensor", &SlicedTensor::GetSlicedTensor)
      .def("__repr__", &SlicedTensor::__repr__)
      .def("Copy_", &SlicedTensor::Copy_, "",
           {torch::arg("right"), torch::arg("non_blocking") = false});

  m.class_<IPCTensorFactory>("IPCTensorFactory")
      .def_static("ClearIPCMemory", &IPCTensorFactory::ClearIPCMemory)
      .def_static("NewIPCTensor", &IPCTensorFactory::NewIPCTensor)
      .def_static("NewIPCGPUTensor", &IPCTensorFactory::NewIPCGPUTensor)
      .def_static("ListIPCTensors", &IPCTensorFactory::ListIPCTensors)
      .def_static("GetIPCTensorFromName",
                  &IPCTensorFactory::GetIPCTensorFromName)
      .def_static("GetSlicedIPCTensorFromName",
                  &IPCTensorFactory::GetSlicedIPCTensorFromName);
}

}  // namespace recstore