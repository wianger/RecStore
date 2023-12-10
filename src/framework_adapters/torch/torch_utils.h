#pragma once
#include <torch/torch.h>

#include <vector>

namespace recstore {
std::string toString(const torch::Tensor &tensor) {
  std::stringstream ss;
  assert(tensor.dim() == 1);
  ss << folly::sformat("[tensor({},{},{},...,)] shape=[{}]",
                       tensor[0].item<int64_t>(), tensor[1].item<int64_t>(),
                       tensor[2].item<int64_t>(), tensor.size(0));
  return ss.str();
}

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