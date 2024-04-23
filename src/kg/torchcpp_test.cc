#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

void test1(torch::Tensor full_emb, int64_t id, torch::Tensor grad) {
  full_emb.index_add_(0, torch::full({1}, id), -grad.cpu());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

TORCH_LIBRARY(librecstore_pytorch_test, m) {
  m.def("test1", &test1);
  ;
}