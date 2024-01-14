#pragma once
#include <torch/torch.h>

#include <vector>

#include "IPCTensor.h"
#include "grad_memory_manager.h"

namespace recstore {

std::string toString(const torch::Tensor &tensor, bool simplified = true);
std::string toString(c10::intrusive_ptr<recstore::SlicedTensor> tensor,
                     bool simplified = true);
std::string toString(const SubGradTensor &tensor, bool simplified = true);

class TensorUtil {
 public:
  static std::vector<torch::Tensor> IndexVectors(
      const torch::Tensor &tensor, const std::vector<torch::Tensor> &indices) {
#ifdef DEBUG
    for (auto each : indices) {
      CHECK(each.dim() == 1);
      // CHECK(each.dtype() == torch::kInt64);
    }
#endif
    std::vector<torch::Tensor> ret;
    ret.reserve(indices.size());
    for (int i = 0; i < indices.size(); i++) {
      // ret.push_back(tensor.index_select(0, indices[i]));
      ret.push_back(at::indexing::get_item(
          tensor, {at::indexing::TensorIndex(indices[i])}));

      // ret.push_back(tensor[indices[i]]);
      // ret.push_back(torch::index_select(tensor, 0, indices));
    }
    return ret;
  }
};
}  // namespace recstore