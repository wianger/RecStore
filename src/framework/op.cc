#include "framework/op.h"

#include <iostream>
#include <stdexcept>
#include <vector>

namespace recstore {

void validate_keys(const base::RecTensor& keys) {
  if (keys.dtype() != base::DataType::UINT64) {
    throw std::invalid_argument("Keys tensor must have dtype UINT64, but got " + base::DataTypeToString(keys.dtype()));
  }
  if (keys.dim() != 1) {
    throw std::invalid_argument(
        "Keys tensor must be 1-dimensional, but has " + std::to_string(keys.dim()) + " dimensions.");
  }
}

void validate_embeddings(const base::RecTensor& embeddings, const std::string& name) {
  if (embeddings.dtype() != base::DataType::FLOAT32) {
    throw std::invalid_argument(
        name + " tensor must have dtype FLOAT32, but got " + base::DataTypeToString(embeddings.dtype()));
  }
  if (embeddings.dim() != 2) {
    throw std::invalid_argument(
        name + " tensor must be 2-dimensional, but has " + std::to_string(embeddings.dim()) + " dimensions.");
  }
  if (embeddings.shape(1) != base::EMBEDDING_DIMENSION_D) {
    throw std::invalid_argument(name + " tensor has embedding dimension " + std::to_string(embeddings.shape(1)) +
                                ", but expected " + std::to_string(base::EMBEDDING_DIMENSION_D));
  }
}

void EmbRead(const base::RecTensor& keys, base::RecTensor& values) {
  validate_keys(keys);
  validate_embeddings(values, "Values");

  const int64_t L = keys.shape(0);
  if (values.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but values has length " +
                                std::to_string(values.shape(0)));
  }

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  float* values_data        = values.data_as<float>();

  // std::cout << "[EmbRead] Reading " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;

  for (int64_t i = 0; i < L; ++i) {
    uint64_t key = keys_data[i];
    // std::cout << "  - Reading embedding for key: " << key << std::endl;

    for (int64_t j = 0; j < base::EMBEDDING_DIMENSION_D; ++j) {
      values_data[i * base::EMBEDDING_DIMENSION_D + j] = static_cast<float>(key % 100) + static_cast<float>(j) * 0.1f;
    }
  }
  // std::cout << "[EmbRead] Read operation complete." << std::endl;
}

void EmbUpdate(const base::RecTensor& keys, const base::RecTensor& grads) {
  validate_keys(keys);
  validate_embeddings(grads, "Grads");

  const int64_t L = keys.shape(0);
  if (grads.shape(0) != L) {
    throw std::invalid_argument("Dimension mismatch: Keys has length " + std::to_string(L) + " but grads has length " +
                                std::to_string(grads.shape(0)));
  }

  const uint64_t* keys_data = keys.data_as<uint64_t>();
  const float* grads_data   = grads.data_as<float>();

  // std::cout << "[EmbUpdate] Updating " << L << " embeddings of dimension " <<
  // base::EMBEDDING_DIMENSION_D << std::endl;
  for (int64_t i = 0; i < L; ++i) {
    uint64_t key             = keys_data[i];
    const float* grad_vector = &grads_data[i * base::EMBEDDING_DIMENSION_D];

    // output for example
    std::cout << "  - Updating embedding for key: " << key << " with first grad element: " << grad_vector[0]
              << std::endl;

    // true_embedding_for_key -= learning_rate * grad_vector;
  }
  // std::cout << "[EmbUpdate] Update operation complete." << std::endl;
}

} // namespace recstore
