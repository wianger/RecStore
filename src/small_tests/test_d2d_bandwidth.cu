#include <cuda_runtime.h>
#include <stdio.h>

#include <chrono>
#include <iostream>

#define SIZE (4 * 1024 * 1024 * 1024LL)

int main() {
    cudaSetDevice(0);
  float* deviceData1;

  cudaMalloc((void**)&deviceData1, SIZE * sizeof(float)
                 );

    cudaSetDevice(1);
  float* deviceData2;
  cudaMalloc((void**)&deviceData2, SIZE * sizeof(float));

    cudaSetDevice(0);
  std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
  cudaMemcpy(deviceData2, deviceData1, SIZE * sizeof(float),
             cudaMemcpyDeviceToDevice);
  cudaDeviceSynchronize();
  std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

  std::cout
      << (double)SIZE / (1024 * 1024 * 1024) * sizeof(float) /
             (double)std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count()
      << " GB/s";

  return 0;
}
