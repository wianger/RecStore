#include "framework/hugectr/op_hugectr.h"
#include "framework/op.h"
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>

static void check_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA Error in op_hugectr.cc: " + std::string(cudaGetErrorString(err)));
  }
}

namespace recstore {
namespace framework {

void emb_read_hugectr(const HugeCTR::Tensor2<long long>& keys,
                        HugeCTR::Tensor2<float>& values) {

  const auto& keys_dims = keys.get_dimensions();
  const auto& values_dims = values.get_dimensions();

  if (keys_dims.size() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional.");
  }
  if (values_dims.size() != 2) {
    throw std::invalid_argument("Values tensor must be 2-dimensional.");
  }
  if (keys_dims[0] != values_dims[0]) {
    throw std::invalid_argument("Keys and Values tensors must have the same size in dimension 0.");
  }
  if (values_dims[1] != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument("Values tensor has incorrect embedding dimension.");
  }

  const int64_t L = keys_dims[0];
  const int64_t D = values_dims[1];

  std::vector<long long> h_keys(L);
  std::vector<float> h_values(L * D);

  check_cuda_error(cudaMemcpy(h_keys.data(), keys.get_ptr(), L * sizeof(long long), cudaMemcpyDeviceToHost));

  base::RecTensor rec_keys(h_keys.data(), {L}, base::DataType::UINT64);
  base::RecTensor rec_values(h_values.data(), {L, D}, base::DataType::FLOAT32);

  try {
    recstore::EmbRead(rec_keys, rec_values);
  } catch (const std::exception& e) {
    throw std::runtime_error("Recstore EmbRead failed during HugeCTR operation: " + std::string(e.what()));
  }

  check_cuda_error(cudaMemcpy(values.get_ptr(), h_values.data(), L * D * sizeof(float), cudaMemcpyHostToDevice));
}


void emb_update_hugectr(const HugeCTR::Tensor2<long long>& keys,
                        const HugeCTR::Tensor2<float>& grads) {

  const auto& keys_dims = keys.get_dimensions();
  const auto& grads_dims = grads.get_dimensions();
  
  if (keys_dims.size() != 1) {
    throw std::invalid_argument("Keys tensor must be 1-dimensional.");
  }
  if (grads_dims.size() != 2) {
    throw std::invalid_argument("Grads tensor must be 2-dimensional.");
  }
  if (keys_dims[0] != grads_dims[0]) {
    throw std::invalid_argument("Keys and Grads tensors must have the same size in dimension 0.");
  }
  if (grads_dims[1] != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument("Grads tensor has incorrect embedding dimension.");
  }

  const int64_t L = keys_dims[0];
  const int64_t D = grads_dims[1];

  std::vector<long long> h_keys(L);
  std::vector<float> h_grads(L * D);

  check_cuda_error(cudaMemcpy(h_keys.data(), keys.get_ptr(), L * sizeof(long long), cudaMemcpyDeviceToHost));
  check_cuda_error(cudaMemcpy(h_grads.data(), grads.get_ptr(), L * D * sizeof(float), cudaMemcpyDeviceToHost));

  base::RecTensor rec_keys(h_keys.data(), {L}, base::DataType::UINT64);
  base::RecTensor rec_grads(h_grads.data(), {L, D}, base::DataType::FLOAT32);

  try {
    recstore::EmbUpdate(rec_keys, rec_grads);
  } catch (const std::exception& e) {
    throw std::runtime_error("Recstore EmbUpdate failed during HugeCTR operation: " + std::string(e.what()));
  }
}

}
}
