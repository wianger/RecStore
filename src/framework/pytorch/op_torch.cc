#include <torch/extension.h>

#include "base/tensor.h"
#include "framework/op.h"

namespace recstore {
namespace framework {
torch::Tensor emb_read_torch(const torch::Tensor& keys) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  const int64_t L = keys.size(0);
  const int64_t D = base::EMBEDDING_DIMENSION_D;

  auto values = torch::empty({L, D}, keys.options().dtype(torch::kFloat32));
  TORCH_CHECK(values.is_contiguous(), "Values tensor must be contiguous");

  base::RecTensor rec_keys(const_cast<void*>(keys.data_ptr()), {L},
                           base::DataType::UINT64);

  base::RecTensor rec_values(values.data_ptr(), {L, D},
                             base::DataType::FLOAT32);
  EmbRead(rec_keys, rec_values);

  return values;
}

void emb_update_torch(const torch::Tensor& keys, const torch::Tensor& grads) {
  TORCH_CHECK(keys.dim() == 1, "Keys tensor must be 1-dimensional");
  TORCH_CHECK(keys.scalar_type() == torch::kInt64,
              "Keys tensor must have dtype int64");
  TORCH_CHECK(keys.is_contiguous(), "Keys tensor must be contiguous");

  TORCH_CHECK(grads.dim() == 2, "Grads tensor must be 2-dimensional");
  TORCH_CHECK(grads.scalar_type() == torch::kFloat32,
              "Grads tensor must have dtype float32");
  TORCH_CHECK(grads.is_contiguous(), "Grads tensor must be contiguous");

  TORCH_CHECK(keys.size(0) == grads.size(0),
              "Keys and Grads tensors must have the same size in dimension 0");
  TORCH_CHECK(grads.size(1) == base::EMBEDDING_DIMENSION_D,
              "Grads tensor has wrong embedding dimension");

  const int64_t L = keys.size(0);
  const int64_t D = grads.size(1);

  base::RecTensor rec_keys(const_cast<void*>(keys.data_ptr()), {L},
                           base::DataType::UINT64);

  base::RecTensor rec_grads(const_cast<void*>(grads.data_ptr()), {L, D},
                            base::DataType::FLOAT32);

  try {
    EmbUpdate(rec_keys, rec_grads);
  } catch (const std::exception& e) {
    throw std::runtime_error("Error in EmbUpdate: " + std::string(e.what()));
  }
}

TORCH_LIBRARY(recstore_ops, m) {
  m.def("emb_read", emb_read_torch);
  m.def("emb_update", emb_update_torch);
}

}  // namespace framework
}  // namespace recstore
