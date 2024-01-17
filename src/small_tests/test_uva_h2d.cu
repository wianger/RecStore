#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

const int cycle = 10000;

__global__ void generatePattern(int *pos, int size) {
  curandState state;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid % 32;
  curand_init(warp_id, 0, 0, &state);
  int *pos_off = pos + warp_id * cycle;
  if (lane == 0) {
    for (int i = 0; i < cycle; i++) {
      int index = curand(&state) % size;
      pos_off[i] = index;
    }
  }
}

__global__ void randomAccessKernel(double4 *data, int size, double4 *output,
                                   int *pos) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid % 32;
  int *pos_off = pos + warp_id * cycle;
  for (int i = 0; i < cycle; i++) {
    int index = pos_off[i];
    output[index + lane] = data[index + lane];
  }
}

int main() {
  const int dataSize = 1024 * 1024 * 256;
  double4 *hostData, *deviceOutput;
  int *pos;
  const int block_size = 512;
  const int thread_cnt = 256 * 256 * 16;
  const int size = dataSize - block_size;

  dim3 blockSize(block_size);
  dim3 gridSize(thread_cnt / blockSize.x);
  int cycle_len = (long)cycle * thread_cnt / 32 * sizeof(int);
  cudaMallocHost(&hostData, dataSize * sizeof(double4));
  cudaMalloc(&deviceOutput, dataSize * sizeof(double4));
  cudaMalloc(&pos, cycle_len);
  for (int i = 0; i < dataSize; ++i) {
    hostData[i] = make_double4(i, i, i, i);
  }
  generatePattern<<<gridSize, blockSize>>>(pos, size);
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  randomAccessKernel<<<gridSize, blockSize>>>(hostData, size, deviceOutput,
                                              pos);

  err = cudaEventRecord(stop);
  cudaError_t err2 = cudaEventSynchronize(stop);
  if (err != cudaSuccess || err2 != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    std::cout << "Error: " << cudaGetErrorString(err2) << std::endl;
  }
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  std::cout << "Execution time: " << milliseconds << " ms" << std::endl;
  std::cout << "bandwidth: "
            << sizeof(double4) * thread_cnt * cycle / milliseconds / 1000000
            << " GB/s" << std::endl;
  cudaFree(hostData);
  cudaFree(deviceOutput);

  return 0;
}
