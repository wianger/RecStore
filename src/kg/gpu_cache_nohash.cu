#include <cstdint>
__global__ static void UVAQueryKernel(const int64_t *d_indices,
                                      const float *d_db, const int emb_dim,
                                      float *d_output_buffer,
                                      const int query_count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < query_count * emb_dim) {
    uint64_t key_idx = idx / emb_dim;
    int sub_idx = idx % emb_dim;
    d_output_buffer[idx] = d_db[d_indices[key_idx] * emb_dim + sub_idx];
  }
}

void UVAQuery(const int64_t *d_indices, const float *d_db, const int emb_dim,
              float *d_output_buffer, const int query_count,
              cudaStream_t stream) {
  uint64_t b_size = 128;
  uint64_t numThreads = query_count * emb_dim;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  UVAQueryKernel<<<g_size, b_size, 1, stream>>>(d_indices, d_db, emb_dim,
                                                d_output_buffer, query_count);
  return;
}