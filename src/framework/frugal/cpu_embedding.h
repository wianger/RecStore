#pragma once

#include <torch/torch.h>

#include <atomic>

#include "base/lock.h"

namespace recstore {
class CPUEmbedding {
 public:
  CPUEmbedding(const std::string &name, torch::Tensor weight)
      : weight_(weight),
        num_embeddings_(weight_.size(0)),
        dim_(weight_.size(1)),
        bitlock_table_(num_embeddings_) {
    ;
  }

  int64_t size(int dim) { return weight_.size(dim); }

  void safe_index_add_(const int64_t index, const at::Tensor &source) {
    // static std::mutex mtx;
    // std::lock_guard<std::mutex> _(mtx);

    CHECK_EQ(source.dim(), 2);
    CHECK_EQ(source.size(1), dim_);
    bitlock_table_.lock(index);
    {
      for (int i = 0; i < dim_; i++) {
        // std::atomic<float> *weight_ptr =
        //     (std::atomic<float> *)&weight_.data_ptr<float>()[index * dim_ +
        //     i];
        // std::atomic<float> *source_ptr =
        //     (std::atomic<float> *)&source.data_ptr<float>()[i];

        // weight_ptr->store(*weight_ptr + *source_ptr);

        weight_.data_ptr<float>()[index * dim_ + i] +=
            source.data_ptr<float>()[i];
      }
      // weight_.index_add_(0, torch::full({1}, index), source);
    }
    bitlock_table_.unlock(index);
  }

  void index_add_(int64_t dim, const at::Tensor &index,
                  const at::Tensor &source) {
    weight_.index_add_(dim, index, source);
  }

  ~CPUEmbedding() = default;

 private:
  torch::Tensor weight_;
  int64_t num_embeddings_;
  int dim_;
  base::BitLockTable bitlock_table_;
};

}  // namespace recstore