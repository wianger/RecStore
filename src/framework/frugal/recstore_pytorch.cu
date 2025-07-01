#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

namespace recstore {
void RegisterKGCacheController(torch::Library &m);

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

}  // namespace recstore