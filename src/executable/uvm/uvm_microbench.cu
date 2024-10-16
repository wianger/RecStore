#include <cuda_runtime.h>
#include <nccl.h>

#include <iostream>

#define CHECK_CUDA(call)                                                   \
  do {                                                                     \
    cudaError_t err = call;                                                \
    if (err != cudaSuccess) {                                              \
      std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ \
                << ": " << cudaGetErrorString(err) << std::endl;           \
      exit(err);                                                           \
    }                                                                      \
  } while (0)

#define CHECK_NCCL(call)                                                   \
  do {                                                                     \
    ncclResult_t res = call;                                               \
    if (res != ncclSuccess) {                                              \
      std::cerr << "NCCL error in " << __FILE__ << " at line " << __LINE__ \
                << ": " << ncclGetErrorString(res) << std::endl;           \
      exit(res);                                                           \
    }                                                                      \
  } while (0)

__global__ void embedding_forward(float* embedding_matrix, uint64_t * indices,
                                  float* output, int V, int D, int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size) {
    int index = indices[idx];
    for (int d = 0; d < D; d++) {
      output[idx * D + d] = embedding_matrix[index * D + d];
    }
  }
}

__global__ void embedding_backward(float* grad_output, int* indices,
                                   float* grad_embedding_matrix, int D,
                                   int batch_size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < batch_size) {
    int index = indices[idx];
    for (int d = 0; d < D; d++) {
      atomicAdd(&grad_embedding_matrix[index * D + d],
                grad_output[idx * D + d]);
    }
  }
}

int main() {
  const int V = 10000;  // 词汇表大小
  const int D = 128;    // 嵌入维度
  const int batch_size = 32;
  const int num_gpus = 2;  // 使用的GPU数量

  // 初始化NCCL
  ncclComm_t comms[num_gpus];
  int devices[num_gpus];
  for (int i = 0; i < num_gpus; i++) {
    devices[i] = i;
  }

  // 创建NCCL communicator
  CHECK_NCCL(ncclCommInitAll(comms, num_gpus, devices));

  // 在每个GPU上分配内存
  float* embedding_matrix[num_gpus];
  float* output[num_gpus];
  float* grad_output[num_gpus];
  float* grad_embedding_matrix[num_gpus];
  int* indices[num_gpus];

  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    CHECK_CUDA(cudaMalloc(&embedding_matrix[i], V * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&output[i], batch_size * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_output[i], batch_size * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&grad_embedding_matrix[i], V * D * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&indices[i], batch_size * sizeof(int)));

    // 初始化embedding矩阵和索引
    for (int j = 0; j < V * D; j++) {
      embedding_matrix[i][j] = static_cast<float>(rand()) / RAND_MAX;
      grad_embedding_matrix[i][j] = 0.0f;
    }

    for (int j = 0; j < batch_size; j++) {
      indices[i][j] = rand() % V;
    }
  }

  // 配置CUDA kernel
  int blockSize = 128;
  int numBlocks = (batch_size + blockSize - 1) / blockSize;

  // 在每个GPU上执行前向传播
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    embedding_forward<<<numBlocks, blockSize>>>(embedding_matrix[i], indices[i],
                                                output[i], V, D, batch_size);
  }

  // 同步所有GPU
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }

#pragma omp parallel for
  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    embedding_backward<<<numBlocks, blockSize>>>(
        grad_output[i], indices[i], grad_embedding_matrix[i], D, batch_size);
    cudaDeviceSynchronize();
    ncclAllReduce((const void*)grad_embedding_matrix[i],
                  (void*)grad_embedding_matrix[0], V * D, ncclFloat, ncclSum,
                  comms[i], 0);
    cudaDeviceSynchronize();
    std::cout << "梯度更新结果:" << std::endl;
    for (int i = 0; i < V * D; i++) {
      if (grad_embedding_matrix[0][i] != 0) {
        std::cout << "grad_embedding_matrix[" << i
                  << "] = " << grad_embedding_matrix[0][i] << std::endl;
      }
    }
  }

  for (int i = 0; i < num_gpus; i++) {
    cudaSetDevice(i);
    cudaFree(embedding_matrix[i]);
    cudaFree(output[i]);
    cudaFree(grad_output[i]);
    cudaFree(grad_embedding_matrix[i]);
    cudaFree(indices[i]);
  }
  for (int i = 0; i < num_gpus; i++) {
    ncclCommDestroy(comms[i]);
  }

  return 0;
}
