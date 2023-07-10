#include <iostream>

#define CUDA_CHECK(val) \
  { nv::cuda_check_((val), __FILE__, __LINE__); }

namespace nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string& what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char* file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}
}  // namespace nv

__global__ void get_and_set_kernel(int* d_a) {
  printf("In card 1: d_a = %d\n", *d_a);
  *d_a = 4321;
}

int main() {
  int* dev_a;
  int host_a = 1234;
  int size = 4;

  int can_access_peer_0_1 = true;
  cudaDeviceCanAccessPeer(&can_access_peer_0_1, 0, 1);
  if (!can_access_peer_0_1) {
    std::cerr << "can not access p2p";
    // std::exit(-1);
  }

  cudaSetDevice(1);
  // 这行是必须的
  cudaDeviceEnablePeerAccess(0, 0);

  cudaSetDevice(0);
  cudaMalloc((void**)&dev_a, size);
  cudaMemcpy(dev_a, &host_a, size, cudaMemcpyHostToDevice);
  CUDA_CHECK(cudaGetLastError());
  //
  cudaSetDevice(1);
  get_and_set_kernel<<<1, 1>>>(dev_a);
  cudaDeviceSynchronize();
  CUDA_CHECK(cudaGetLastError());
  //
  cudaSetDevice(0);
  cudaMemcpy(&host_a, dev_a, size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  std::cout << "In card 0: d_a = " << host_a << "\n";
  CUDA_CHECK(cudaGetLastError());

  return 0;
}