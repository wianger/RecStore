#pragma once
#include <cuda.h>

#include <stdexcept>
#include <string>

namespace xmh {
class cudaEventWrapper {
  cudaEvent_t event;

 public:
  cudaEventWrapper(cudaStream_t stream = 0) {
    auto err = cudaEventCreateWithFlags(&event, cudaEventDefault);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("Failed to create event: ") +
                               cudaGetErrorString(err));
    }

    err = cudaEventRecord(event, stream);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("Failed to record event on stream: ") +
          cudaGetErrorString(err));
    }
  }

  ~cudaEventWrapper() { cudaEventDestroy(event); }

  double operator-(const cudaEventWrapper& other) const {
    float msecs = 0;
    auto err = cudaEventElapsedTime(&msecs, other.event, event);
    if (err != cudaSuccess) {
      throw std::runtime_error(
          std::string("Could not calculate elapsed time: ") +
          cudaGetErrorString(err));
    }

    return ((double)msecs) * 1e3;
  }
};

}  // namespace xmh