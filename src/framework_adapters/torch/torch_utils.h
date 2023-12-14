#pragma once
#include <torch/torch.h>

#include <vector>

namespace recstore {
template <typename T>
std::string toStringInner(const torch::Tensor &tensor, bool simplified = true) {
  std::stringstream ss;
  if (tensor.dim() == 1) {
    if (simplified) {
      // ss << folly::sformat("[tensor({},{},{},...,)] shape=[{}]",
      //                      tensor[0].item<T>(), tensor[1].item<T>(),
      //                      tensor[2].item<T>(), tensor.size(0));
      ss << folly::sformat("[tensor({},{},...,)] shape=[{}]",
                           tensor[0].item<T>(), tensor[1].item<T>(),
                           tensor.size(0));
    } else {
      ss << "tensor([";
      for (int i = 0; i < tensor.size(0); ++i) {
        ss << folly::sformat("{},", tensor[i].item<T>());
      }
      ss << "])";
    }
  } else if (tensor.dim() == 2) {
    ss << "tensor([";
    for (int i = 0; i < tensor.size(0); i++) {
      ss << "[";
      for (int j = 0; j < tensor.size(1); j++) {
        ss << folly::sformat("{},", tensor[i][j].item<T>());
      }
      ss << "],";
    }
    ss << "])";
  } else {
    assert(0);
  }
  return ss.str();
}

std::string toString(const torch::Tensor &tensor, bool simplified = true) {
  if (tensor.scalar_type() == torch::kFloat32)
    return toStringInner<float>(tensor, simplified);
  else if (tensor.scalar_type() == torch::kInt64)
    return toStringInner<int64_t>(tensor, simplified);
  else
    LOG(FATAL) << "to String, not supported type";
}

std::string toString(c10::intrusive_ptr<recstore::SlicedTensor> tensor,
                     bool simplified = true) {
  return toString(tensor->GetSlicedTensor(), simplified);
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