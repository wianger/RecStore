#pragma once

#include <string>

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
class CudaDeviceRestorer {
 public:
  CudaDeviceRestorer() { CUDA_CHECK(cudaGetDevice(&dev_)); }
  ~CudaDeviceRestorer() { CUDA_CHECK(cudaSetDevice(dev_)); }
  void check_device(int device) const {
    if (device != dev_) {
      throw std::runtime_error(std::string(__FILE__) + ":" +
                               std::to_string(__LINE__) +
                               ": Runtime Error: The device id in the context "
                               "is not consistent with configuration");
    }
  }

 private:
  int dev_;
};
}  // namespace nv
