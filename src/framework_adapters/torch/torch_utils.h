#pragma once

#include <torch/torch.h>

namespace recstore {
std::string toString(const torch::Tensor &tensor) {
  std::stringstream ss;
  assert(tensor.dim() == 1);
  ss << folly::sformat("[tensor({},{},{},...,)] shape=[{}]",
                       tensor[0].item<int64_t>(), tensor[1].item<int64_t>(),
                       tensor[2].item<int64_t>(), tensor.size(0));
  return ss.str();
}
}  // namespace recstore