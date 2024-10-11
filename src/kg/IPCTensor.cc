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
  _mm_mfence();
  IPCTensorMemoryHandle *handle = IPCMemory::GetInstance()->GetHandle(name);
  if (nullptr == handle) {
    LOG(FATAL) << "IPCTensor " << name << " not found";
  }
  return c10::make_intrusive<SlicedTensor>(handle);
}

c10::intrusive_ptr<SlicedTensor> IPCTensorFactory::NewSlicedIPCGPUTensor(
    const std::string &name, const at::IntArrayRef shape,
    const at::ScalarType dtype, const int64_t dev_id) {
  _mm_mfence();
  IPCTensorFactory::NewIPCGPUTensor(name, shape, dtype, dev_id);
  return IPCTensorFactory::GetSlicedIPCTensorFromName(name);
}

c10::intrusive_ptr<SlicedTensor> IPCTensorFactory::NewSlicedIPCTensor(
    const std::string &name, const at::IntArrayRef shape,
    const at::ScalarType dtype) {
  _mm_mfence();
  IPCTensorFactory::NewIPCTensor(name, shape, dtype);
  return IPCTensorFactory::GetSlicedIPCTensorFromName(name);
}

class Mfence: public torch::CustomClassHolder {
 public:
  static void mfence() { asm volatile("mfence" ::: "memory"); }
  static void lfence() { asm volatile("lfence" ::: "memory"); }
  static void sfence() { asm volatile("sfence" ::: "memory"); }
  static void complier_barrier() { asm volatile("" ::: "memory"); }
};

void RegisterIPCTensorFactory(torch::Library &m) {
  m.class_<SlicedTensor>("SlicedTensor")
      .def("GetSlicedTensor", &SlicedTensor::GetSlicedTensor)
      .def("__repr__", &SlicedTensor::__repr__)
      .def("SetSameShape4Debug", &SlicedTensor::SetSameShape4Debug)
      .def("Copy_", &SlicedTensor::Copy_, "",
           {torch::arg("right"), torch::arg("non_blocking") = false});

  m.class_<IPCTensorFactory>("IPCTensorFactory")
      .def_static("ClearIPCMemory", &IPCTensorFactory::ClearIPCMemory)
      .def_static("NewIPCTensor", &IPCTensorFactory::NewIPCTensor)
      .def_static("NewOrGetIPCTensor", &IPCTensorFactory::NewOrGetIPCTensor)
      .def_static("NewIPCGPUTensor", &IPCTensorFactory::NewIPCGPUTensor)
      .def_static("ListIPCTensors", &IPCTensorFactory::ListIPCTensors)
      .def_static("FindIPCTensorFromName",
                  &IPCTensorFactory::FindIPCTensorFromName)
      .def_static("GetSlicedIPCTensorFromName",
                  &IPCTensorFactory::GetSlicedIPCTensorFromName)
      .def_static("NewSlicedIPCTensor", &IPCTensorFactory::NewSlicedIPCTensor)
      .def_static("NewSlicedIPCGPUTensor",
                  &IPCTensorFactory::NewSlicedIPCGPUTensor);

  m.class_<Mfence>("Mfence")
      .def_static("mfence", &Mfence::mfence)
      .def_static("lfence", &Mfence::lfence)
      .def_static("sfence", &Mfence::sfence);
}

}  // namespace recstore