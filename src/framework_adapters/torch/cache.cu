#include "storage/gpu_cache/nv_gpu_cache.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace recstore {

struct CacheQueryResult : public torch::CustomClassHolder {
  CacheQueryResult(torch::Tensor values, torch::Tensor missing_index,
                   torch::Tensor missing_keys)
      : values_(values),
        missing_index_(missing_index),
        missing_keys_(missing_keys) {}

  torch::Tensor values_;
  torch::Tensor missing_index_;
  torch::Tensor missing_keys_;
};

// template <typename key_t = uint64_t>
class GpuCache : public torch::CustomClassHolder {
  using key_t = uint64_t;
  constexpr static int set_associativity = 2;
  constexpr static int WARP_SIZE = 32;
  constexpr static int bucket_size = WARP_SIZE * set_associativity;

 public:
  GpuCache(int64_t num_items, int64_t num_feats)
      : num_feats(num_feats),
        cache((num_items + bucket_size - 1) / bucket_size, num_feats) {}

  c10::intrusive_ptr<CacheQueryResult> Query(torch::Tensor keys) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    torch::Device device = keys.device();
    CHECK(device.is_cuda())
        << "The tensor of requested indices must be on GPU.";

    // torch::Device device(torch::kCUDA, 0);

    torch::Tensor values = at::zeros(
        torch::IntArrayRef({(int64_t)keys.sizes()[0], (int64_t)num_feats}),
        at::TensorOptions().dtype(torch::kFloat).device(device));

    torch::Tensor missing_index =
        torch::zeros(torch::IntArrayRef({(int64_t)keys.sizes()[0]}),
                     at::TensorOptions().dtype(torch::kLong).device(device));

    torch::Tensor missing_keys = torch::zeros_like(keys);
    torch::Tensor missing_len = torch::zeros(1, torch::kLong);

    // cache.Query(static_cast<const key_t *>(keys.data_ptr<key_t>()),
    //             keys.sizes()[0], static_cast<float
    //             *>(values.data_ptr<float>()), static_cast<uint64_t
    //             *>(missing_index.data_ptr<uint64_t>()), static_cast<key_t
    //             *>(missing_keys.data_ptr<key_t>()),
    //             missing_len.data_ptr<uint64_t>(), stream);

    torch::Tensor missing_len_host = missing_len.cpu();

    TORCH_CHECK_GE(missing_len_host.item<int64_t>(), 0);
    TORCH_CHECK_LE(missing_len_host.item<int64_t>(), keys.sizes()[0]);

    // missing_index = missing_index.CreateView({(int64_t)missing_len_host},
    //                                          missing_index->dtype);
    // missing_keys =
    //     missing_keys.CreateView({(int64_t)missing_len_host},
    //     keys->dtype);

    return c10::make_intrusive<CacheQueryResult>(values, missing_index,
                                                 missing_keys);
  }

  //   void Replace(IdArray keys, NDArray values) {
  //     cudaStream_t stream = dgl::runtime::getCurrentCUDAStream();
  //     CHECK_EQ(keys->shape[0], values->shape[0])
  //         << "First dimensions of keys and values must match";
  //     CHECK_EQ(values->shape[1], num_feats) << "Embedding dimension must
  //     match"; cache.Replace(static_cast<const key_t *>(keys->data),
  //     keys->shape[0],
  //                   static_cast<const float *>(values->data), stream);
  //   }

  virtual ~GpuCache() = default;

 private:
  size_t num_feats;
  gpu_cache::gpu_cache<key_t, uint64_t, std::numeric_limits<key_t>::max(),
                       set_associativity, WARP_SIZE>
      cache;
};

TORCH_LIBRARY(librecstore_pytorch, m) {
  // m.class_<GpuCache>("GpuCache")
  //     .def(torch::init<int64_t, int64_t>())
  //     .def("Query", &GpuCache::Query);
  m.class_<GpuCache>("GpuCache").def(torch::init<int64_t, int64_t>());
}

}  // namespace recstore