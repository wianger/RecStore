#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include "IPCTensor.h"
#include "base/timer.h"
#include "folly/init/Init.h"
#include "gpu_cache_nohash.h"
#include "storage/gpu_cache/nv_gpu_cache.hpp"

namespace recstore {
void RegisterKGCacheController(torch::Library &m);

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

  c10::intrusive_ptr<CacheQueryResult> Query(torch::Tensor keys,
                                             torch::Tensor values) {
    // std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
    // torch::Tensor keys) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    torch::Device device = keys.device();
    TORCH_CHECK(device.is_cuda(),
                "The tensor of requested indices must be on GPU.");

    TORCH_CHECK(keys.scalar_type() == torch::kLong,
                "The tensor of requested indices must be of type int64.");

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

    TORCH_CHECK(missing_len_host >= 0, "");
    TORCH_CHECK(missing_len_host <= keys.sizes()[0], "");

    torch::Tensor ret_missing_index =
        missing_index.slice(0, 0, missing_len_host);
    torch::Tensor ret_missing_key = missing_keys.slice(0, 0, missing_len_host);

    return c10::make_intrusive<CacheQueryResult>(values, ret_missing_index,
                                                 ret_missing_key);
    // return std::make_tuple(values, ret_missing_index, ret_missing_key);
  }

  void Replace(torch::Tensor keys, torch::Tensor values) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();
    TORCH_CHECK(keys.scalar_type() == torch::kLong,
                "The tensor of requested indices must be of type int64.");
    TORCH_CHECK(keys.sizes()[0] == values.sizes()[0],
                "First dimensions of keys and values must match");
    TORCH_CHECK(values.sizes()[1] == emb_dim,
                "Embedding dimension must match ");

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

__global__ void uva_cache_query_kernel(
    float *merge_dst, const int64_t *id_tensor, const float *hbm_tensor,
    float *dram_tensor, const int64_t cached_start_key,
    const int64_t cached_end_key, const size_t len, const size_t emb_vec_size,
    const size_t dram_tensor_size, const size_t hbm_tensor_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= (len * emb_vec_size)) {
    return;
  }

  size_t emb_idx = idx / emb_vec_size;
  size_t float_idx = idx % emb_vec_size;

  int64_t key = id_tensor[emb_idx];

  if (key < cached_start_key || key >= cached_end_key) {
    assert(key * emb_vec_size + float_idx < dram_tensor_size);
    assert(key * emb_vec_size + float_idx >= 0);
    merge_dst[idx] = dram_tensor[key * emb_vec_size + float_idx];
  } else {
    key -= cached_start_key;
    assert(key * emb_vec_size + float_idx < hbm_tensor_size);
    assert(key * emb_vec_size + float_idx >= 0);
    merge_dst[idx] = hbm_tensor[key * emb_vec_size + float_idx];
  }
}
void uva_cache_query_op(at::Tensor merge_dst, const at::Tensor id_tensor,
                        const at::Tensor hbm_tensor,
                        const at::Tensor dram_tensor,
                        const long cached_start_key,
                        const long cached_end_key) {
  // std::cout << "called uva_cache_query_op" << std::endl << std::flush;
  const size_t BLOCK_SIZE = 256;
  const size_t emb_vec_size = merge_dst.size(1);
  const size_t len = merge_dst.size(0);
  TORCH_CHECK(merge_dst.size(0) == id_tensor.size(0),
              "len(merge_dst)!=len(id_tensor)");
  TORCH_CHECK(id_tensor.dtype() == at::kLong, "id_tensor must be int64");
  TORCH_CHECK(hbm_tensor.size(0) == cached_end_key - cached_start_key,
              "len(hbm_tensor) != end-start");

  const size_t len_in_float = len * emb_vec_size;
  const size_t num_blocks = (len_in_float - 1) / BLOCK_SIZE + 1;

  uva_cache_query_kernel<<<num_blocks, BLOCK_SIZE, 0,
                           at::cuda::getCurrentCUDAStream()>>>(
      merge_dst.data_ptr<float>(), id_tensor.data_ptr<int64_t>(),
      hbm_tensor.data_ptr<float>(), dram_tensor.data_ptr<float>(),
      cached_start_key, cached_end_key, len, emb_vec_size, dram_tensor.numel(),
      hbm_tensor.numel());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// at::Tensor renumberingGraphID(const at::Tensor originalID,
//                               const at::Tensor cached_keys,
//                               const int64_t num_gid) {
//   TORCH_CHECK(originalID.dtype() == at::kLong, "");
//   TORCH_CHECK(cached_keys.dtype() == at::kLong, "");

//   at::Tensor ret = originalID.clone();
//   int64_t originalIDPtr = originalID.data_ptr<int64_t>();
//   int length = originalID.numel();

//   int num_cached_keys = cached_keys.numel();
//   std::unordered_map<int64_t, int64_t> renumbering_map;
//   renumbering_map.reserve(length);

//   for (int i = 0; i < num_cached_keys; i++) {
//     renumbering_map[cached_keys[i]] = i;
//   }

//   std::atomic<int64_t>

//       return ret;
// }

void init_folly() {
  std::vector<std::string> arguments = {"program_name", "--logtostderr"};
  int argc = static_cast<int>(arguments.size());
  char **argv = new char *[argc];
  for (int i = 0; i < argc; ++i) {
    argv[i] = new char[arguments[i].size() + 1];
    std::strcpy(argv[i], arguments[i].c_str());
  }
  folly::init(&argc, (char ***)&argv, false);
  xmh::Reporter::StartReportThread();
}

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
  m.def("uva_cache_query_op", &uva_cache_query_op);
  m.def("init_folly", &init_folly);

  m.class_<GPUCacheWithNoHashTorch>("GpuCacheWithNoHash")
      .def(torch::init<int64_t, int64_t, int64_t, int64_t>());
  RegisterIPCTensorFactory(m);
  RegisterKGCacheController(m);
}

}  // namespace recstore