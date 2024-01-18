#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <iostream>

const int cycle = 1;

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

// __global__ void randomAccessKernel(double4 *data, int size, double4 *output,
//                                    int *pos) {
//   int tid = blockIdx.x * blockDim.x + threadIdx.x;
//   int warp_id = tid / 32;
//   int lane = tid % 32;
//   int *pos_off = pos + warp_id * cycle;
//   for (int i = 0; i < cycle; i++) {
//     int index = pos_off[i];
//     output[index + lane] = data[index + lane];
//   }
// }

__global__ void randomAccessKernel(float *src, int size_dim, float *dst) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int warp_id = tid / 32;
  int lane = tid % 32;

  if (tid < size_dim) dst[tid] = src[tid];
}

int main() {
  const int start_dim = 1 * 1024 / sizeof(float);
  const int end_dim = 128 * 1024 * 1024 / sizeof(float);

  float *hostData, *deviceOutput;
  cudaMallocHost(&hostData, end_dim * sizeof(float));
  cudaMalloc(&deviceOutput, end_dim * sizeof(float));

  for (int nr_dim = start_dim; nr_dim <= end_dim; nr_dim *= 2) {
    const int block_size = 256;
    const int block_cnt = nr_dim / block_size + 1;

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    randomAccessKernel<<<block_cnt, block_size>>>(hostData, nr_dim,
                                                  deviceOutput);
    err = cudaEventRecord(stop);
    cudaError_t err2 = cudaEventSynchronize(stop);

    if (err != cudaSuccess || err2 != cudaSuccess) {
      std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
      std::cout << "Error: " << cudaGetErrorString(err2) << std::endl;
    }
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << nr_dim * sizeof(float) << "\t"
              << sizeof(float) * nr_dim / milliseconds / 1000000 << " GB/s"
              << std::endl;
  }

  return 0;
}
