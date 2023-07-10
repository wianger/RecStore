#include "storage/gpu_cache/nv_gpu_cache.hpp"

#include "gpu_cache_nohash.h"

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

  std::string __repr__() const {
    std::stringstream ss;
    ss << "CacheQueryResult(values=" << values_
       << ", missing_index=" << missing_index_
       << ", missing_keys=" << missing_keys_ << ")";
    return ss.str();
  }

  torch::Tensor values() { return values_; }
  torch::Tensor missing_index() { return missing_index_; }
  torch::Tensor missing_keys() { return missing_keys_; }

  torch::Tensor values_;
  torch::Tensor missing_index_;
  torch::Tensor missing_keys_;
};

// template <typename key_t = uint64_t>
class GpuCache : public torch::CustomClassHolder {
  using key_t = int64_t;
  constexpr static int set_associativity = 2;
  constexpr static int WARP_SIZE = 32;
  constexpr static int bucket_size = WARP_SIZE * set_associativity;

 public:
  GpuCache(int64_t num_items, int64_t emb_dim)
      : emb_dim(emb_dim),
        cache((num_items + bucket_size - 1) / bucket_size, emb_dim) {}

  c10::intrusive_ptr<CacheQueryResult> Query(torch::Tensor keys, torch::Tensor values) {
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
    // torch::Tensor keys) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    torch::Device device = keys.device();
    TORCH_CHECK(device.is_cuda(),
                "The tensor of requested indices must be on GPU.");

    TORCH_CHECK_EQ(keys.scalar_type(), torch::kLong)
        << "The tensor of requested indices must be of type int64.";

    // torch::Tensor values = at::zeros(
    //     torch::IntArrayRef({(int64_t)keys.sizes()[0], (int64_t)emb_dim}),
    //     at::TensorOptions().dtype(torch::kFloat).device(device));

    torch::Tensor missing_index =
        torch::zeros(torch::IntArrayRef({(int64_t)keys.sizes()[0]}),
                     at::TensorOptions().dtype(torch::kLong).device(device));

    torch::Tensor missing_keys = torch::zeros_like(keys);
    torch::Tensor missing_len =
        at::zeros(1, at::TensorOptions().dtype(torch::kLong).device(device));

    cache.Query(static_cast<const key_t *>(keys.data_ptr<key_t>()),
                keys.sizes()[0], static_cast<float *>(values.data_ptr<float>()),
                (uint64_t *)(missing_index.data_ptr<int64_t>()),
                static_cast<key_t *>(missing_keys.data_ptr<key_t>()),
                (uint64_t *)missing_len.data_ptr<int64_t>(), stream);

    auto missing_len_host = missing_len.cpu().item<int64_t>();

    TORCH_CHECK_GE(missing_len_host, 0);
    TORCH_CHECK_LE(missing_len_host, keys.sizes()[0]);

    torch::Tensor ret_missing_index =
        missing_index.slice(0, 0, missing_len_host);
    torch::Tensor ret_missing_key = missing_keys.slice(0, 0, missing_len_host);

    return c10::make_intrusive<CacheQueryResult>(values, ret_missing_index,
                                                 ret_missing_key);
    // return std::make_tuple(values, ret_missing_index, ret_missing_key);
  }

  void Replace(torch::Tensor keys, torch::Tensor values) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    TORCH_CHECK_EQ(keys.scalar_type(), torch::kLong)
        << "The tensor of requested indices must be of type int64.";
    TORCH_CHECK_EQ(keys.sizes()[0], values.sizes()[0])
        << "First dimensions of keys and values must match";
    TORCH_CHECK_EQ(values.sizes()[1], emb_dim)
        << "Embedding dimension must match ";

    cache.Replace(keys.data_ptr<key_t>(), keys.sizes()[0],
                  values.data_ptr<float>(), stream);
  }

  virtual ~GpuCache() = default;

 private:
  size_t emb_dim;
  gpu_cache::gpu_cache<key_t, uint64_t, std::numeric_limits<key_t>::max(),
                       set_associativity, WARP_SIZE>
      cache;
};

void merge_op(at::Tensor merge_dst, const at::Tensor retrieved,
              const at::Tensor missing_index);

TORCH_LIBRARY(librecstore_pytorch, m) {
  m.class_<CacheQueryResult>("CacheQueryResult")
      .def("__str__", &CacheQueryResult::__repr__)
      .def_property("values", &CacheQueryResult::values)
      .def_property("missing_index", &CacheQueryResult::missing_index)
      .def_property("missing_keys", &CacheQueryResult::missing_keys);

  m.class_<GpuCache>("GpuCache")
      .def(torch::init<int64_t, int64_t>())
      .def("Query", &GpuCache::Query)
      .def("Replace", &GpuCache::Replace);
  
  m.def("merge_op", &merge_op);


  m.class_<GPUCacheWithNoHashTorch>("GpuCacheWithNoHash")
    .def(torch::init<int64_t, int64_t, int64_t,int64_t>());
}

}  // namespace recstore