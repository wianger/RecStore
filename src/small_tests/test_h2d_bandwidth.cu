#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <iostream>

int main() {
  int64_t start = 1 * 1024;
  int64_t end = 128 * 1024 * 1024;

  float* hostData;
  cudaMallocHost((void**)&hostData, end, cudaHostAllocDefault);
  float* deviceData;
  cudaMalloc((void**)&deviceData, end);
  if (hostData == NULL) {
    printf("无法分配主机内存\n");
    return 1;
  }

  for (int64_t each = start; each <= end; each *= 2) {
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cudaMemcpy(deviceData, hostData, each, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

    std::chrono::duration<double> diff = t2 - t1;

    double s_avg = (double)diff.count();

    std::cout << each << "\t" << ((double)each) / s_avg / (1024 * 1024 * 1024LL)
              << " GB/s\n";
  }

  cudaFreeHost(hostData);
  return 0;
}
