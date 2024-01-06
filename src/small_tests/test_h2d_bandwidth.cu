#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <iostream>

#define SIZE (4 * 1024 * 1024 * 1024LL)

int main() {
  // 分配主机内存
  float* hostData;

  cudaMallocHost((void**)&hostData, SIZE * sizeof(float), cudaHostAllocDefault);

  if (hostData == NULL) {
    printf("无法分配主机内存\n");
    return 1;
  }

  // 分配设备内存
  float* deviceData;
  cudaMalloc((void**)&deviceData, SIZE * sizeof(float));

  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

  // 执行 cudaMemcpy 操作
  for (int i = 0; i < 2; i++)
    cudaMemcpy(deviceData, hostData, SIZE * sizeof(float),
               cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  std::cout
      << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

  // 释放内存

  return 0;
}
