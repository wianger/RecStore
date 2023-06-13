#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

__global__ void merge_emb_vec(float *d_output_emb_vec,
                              const float *d_missing_emb_vec,
                              const int64_t *d_missing_index, const size_t len,
                              const size_t emb_vec_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < (len * emb_vec_size)) {
    size_t src_emb_vec = idx / emb_vec_size;
    size_t dst_emb_vec = d_missing_index[src_emb_vec];
    size_t dst_float = idx % emb_vec_size;
    d_output_emb_vec[dst_emb_vec * emb_vec_size + dst_float] =
        d_missing_emb_vec[src_emb_vec * emb_vec_size + dst_float];
  }
}

void merge_emb_vec_async(float *d_vals_merge_dst_ptr,
                         const float *d_vals_retrieved_ptr,
                         const int64_t *d_missing_index_ptr,
                         const size_t missing_len, const size_t emb_vec_size,
                         const size_t BLOCK_SIZE, cudaStream_t stream) {
  if (missing_len == 0) {
    return;
  }
  size_t missing_len_in_float = missing_len * emb_vec_size;
  merge_emb_vec<<<((missing_len_in_float - 1) / BLOCK_SIZE) + 1, BLOCK_SIZE, 0,
                  stream>>>(d_vals_merge_dst_ptr, d_vals_retrieved_ptr,
                            d_missing_index_ptr, missing_len, emb_vec_size);
}

namespace recstore {

void merge_op(at::Tensor merge_dst, const at::Tensor retrieved,
              const at::Tensor missing_index) {
  const size_t missing_len = missing_index.size(0);
  const size_t emb_vec_size = retrieved.size(1);

  TORCH_CHECK(merge_dst.size(1) == emb_vec_size);
  TORCH_CHECK(retrieved.size(1) == emb_vec_size);
  TORCH_CHECK(retrieved.size(0) == missing_index.size(0));

  TORCH_CHECK(retrieved.is_cuda());
  TORCH_CHECK(missing_index.is_cuda());

  const size_t BLOCK_SIZE = 256;
  const size_t missing_len_in_float = missing_len * emb_vec_size;
  const size_t num_blocks = (missing_len_in_float - 1) / BLOCK_SIZE + 1;
  merge_emb_vec_async(merge_dst.data_ptr<float>(), retrieved.data_ptr<float>(),
                      missing_index.data_ptr<int64_t>(), missing_len,
                      emb_vec_size, BLOCK_SIZE,
                      at::cuda::getCurrentCUDAStream());
  // AT_CUDA_CHECK(cudaGetLastError());
  // AT_CUDA_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
}
}  // namespace recstore