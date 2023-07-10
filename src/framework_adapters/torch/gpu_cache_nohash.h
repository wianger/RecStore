#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <cinttypes>
#include <memory>

#include "shm_helper.h"

typedef struct shmStruct_st {
  static constexpr int MAX_DEVICE = 8;
  cudaIpcMemHandle_t memHandle[MAX_DEVICE];
} shmStruct;

class CUDAIPCEnv {
 public:
  CUDAIPCEnv(int32_t device_count) {
    std::cout << "start initialize ipc env\n";
    const char shmName[] = "simpleIPCshm";
    if (sharedMemoryCreate(shmName, sizeof(*shm_), &info_) != 0) {
      std::cout << "Failed to create shared memory slab" << std::endl;
      exit(-1);
    }
    std::cout << "Shared Memory Opened\n";

    shm_ = (volatile shmStruct *)info_.addr;
    memset((void *)shm_, 0, sizeof(*shm_));

    device_count_ = device_count;
  }

 private:
  volatile shmStruct *shm_;
  int32_t device_count_;
  sharedMemoryInfo info_;
};

void UVAQuery(const int64_t *d_indices, const float *d_db, const int emb_dim,
              float *d_output_buffer, const int query_count,
              cudaStream_t stream);

template <typename ID_Type = int64_t>
class GPUCacheWithNoHash {
 public:
  GPUCacheWithNoHash(int64_t capacity, int emb_dim, ID_Type start, ID_Type end,
                     float *d_cache_db)
      : capacity_(capacity),
        emb_dim_(emb_dim),
        start_(start),
        end_(end),
        d_cache_db_(d_cache_db) {
    CHECK_EQ(end - start, capacity);
    // if (nullptr == d_cache_db_) {
    //   cudaMalloc(&d_cache_db_, capacity * emb_dim * sizeof(float));
    // }
    CUDA_CHECK(cudaGetLastError());
  }

  void Query(const ID_Type *d_indices, int key_len, float *d_value,
             cudaStream_t stream) {
    UVAQuery(d_indices, d_cache_db_, emb_dim_, d_value, key_len, stream);
    CUDA_CHECK(cudaGetLastError());
  }

 private:
  const int64_t capacity_;
  const int emb_dim_;
  const ID_Type start_;
  const ID_Type end_;
  float *d_cache_db_;
};

template <typename ID_Type = int64_t>
class MultiGPUCacheWithNoHash {
 public:
  MultiGPUCacheWithNoHash(int64_t capacity, int emb_dim, ID_Type start, ID_Type end,
                     float *d_cache_db)
      : capacity_(capacity),
        emb_dim_(emb_dim),
        start_(start),
        end_(end),
        d_cache_db_(d_cache_db) {
    CHECK_EQ(end - start, capacity);
    // if (nullptr == d_cache_db_) {
    //   cudaMalloc(&d_cache_db_, capacity * emb_dim * sizeof(float));
    // }
    CUDA_CHECK(cudaGetLastError());
  }

  void Query(const ID_Type *d_indices, int key_len, float *d_value,
             cudaStream_t stream) {
    UVAQuery(d_indices, d_cache_db_, emb_dim_, d_value, key_len, stream);
    CUDA_CHECK(cudaGetLastError());
  }

 private:
  const int64_t capacity_;
  const int emb_dim_;
  const ID_Type start_;
  const ID_Type end_;
  float *d_cache_db_;
};




class GPUCacheWithNoHashTorch : public torch::CustomClassHolder {
 public:
  GPUCacheWithNoHashTorch(int64_t capacity, int64_t emb_dim, int64_t start,
                          int64_t end) {
    cache_ = std::make_unique<GPUCacheWithNoHash<int64_t>>(capacity, (int)emb_dim, start,
                                                  end, nullptr);
  }

  void Query(torch::Tensor keys, torch::Tensor values) {
    const cudaStream_t stream = at::cuda::getDefaultCUDAStream();

    torch::Device device = keys.device();
    TORCH_CHECK(device.is_cuda(),
                "The tensor of requested indices must be on GPU.");
    TORCH_CHECK_EQ(keys.scalar_type(), torch::kLong)
        << "The tensor of requested indices must be of type int64.";

    cache_->Query(keys.data_ptr<int64_t>(), keys.size(0),
                  values.data_ptr<float>(), stream);
  }

 private:
  std::unique_ptr<GPUCacheWithNoHash<int64_t>> cache_;
};