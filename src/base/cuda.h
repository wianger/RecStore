#pragma once

#define XMH_CUDA_CHECK(val) \
  { xmh_nv::cuda_check_((val), __FILE__, __LINE__); }

namespace xmh_nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}

struct Event {
  cudaEvent_t event;

  inline Event(cudaStream_t stream = 0) {
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

  inline ~Event() { cudaEventDestroy(event); }

  inline double operator-(const Event &other) const {
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

}  // namespace xmh_nv
