#pragma once

#include <cuda.h>
#include <cuda_runtime_api.h>
#include "folly/GLog.h"
#include "gdrapi.h"
#include <mutex>
#include <assert.h>

#define ASSERTDRV(stmt)                               \
  do {                                                \
    CUresult result = (stmt);                         \
    if (result != CUDA_SUCCESS) {                     \
      const char *_err_name;                          \
      cuGetErrorName(result, &_err_name);             \
      fprintf(stderr, "CUDA error: %s\n", _err_name); \
    }                                                 \
    assert(CUDA_SUCCESS == result);                   \
  } while (0)

#define XMH_CUDA_CHECK(val) \
  { xmh_nv::cuda_check_((val), __FILE__, __LINE__); }

namespace xmh_nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) + ": CUDA error " +
                        std::to_string(val) + ": " + cudaGetErrorString(val));
  }
}
}  // namespace xmh_nv

class MappingDeviceMemory {
 public:
  static MappingDeviceMemory *GetInstance() {
    static MappingDeviceMemory *mapping_device_memory = nullptr;
    static std::mutex m;
    if (!mapping_device_memory) {
      std::lock_guard<std::mutex> _(m);
      if (!mapping_device_memory) {
        mapping_device_memory = new MappingDeviceMemory();
      }
    }
    return mapping_device_memory;
  }

  struct MetaAlloc {
    void *dev_ptr = nullptr;
    void *mapping_ptr = nullptr;
    gdr_mh_t memory_handle;
  };

  constexpr static int kAlignBytes = 256;

  constexpr bool is_pow2(std::size_t n) {
    return (0 == (n & (n - 1)));
  }
  constexpr bool is_supported_alignment(std::size_t alignment) {
    return is_pow2(alignment);
  }
  constexpr std::size_t align_up(std::size_t v, std::size_t align_bytes) noexcept {
    assert(is_supported_alignment(align_bytes));
    return (v + (align_bytes - 1)) & ~(align_bytes - 1);
  }
  constexpr std::size_t align_down(std::size_t v, std::size_t align_bytes) noexcept {
    assert(is_supported_alignment(align_bytes));
    return v & ~(align_bytes - 1);
  }

  MetaAlloc allocate(size_t size) {
    size = align_up(size, kAlignBytes);
    auto offset_old = offset_.fetch_add(size);
    CHECK_LE(offset_.load(), size_);
    void *dev_ptr = (char *)dev_base_ptr_ + offset_old;
    void *mapping_ptr = (char *)map_base_ptr_ + offset_old;
    return MetaAlloc{
        .dev_ptr = dev_ptr, .mapping_ptr = mapping_ptr, .memory_handle = memory_handle_};
  }

  ~MappingDeviceMemory() {
    CHECK_EQ(0, gdr_unmap(g_, memory_handle_, map_d_ptr, size_));
    CHECK_EQ(0, gdr_unpin_buffer(g_, memory_handle_));
    CHECK_EQ(0, gdr_close(g_));
    XMH_CUDA_CHECK(cudaFree(cudaMallocPtr_));
  }

 private:
  std::atomic<size_t> offset_{0};
  void *dev_base_ptr_ = nullptr;
  void *map_base_ptr_ = nullptr;   // used for mapping copy to/from
  void *map_d_ptr = nullptr;       // used for mapping
  void *cudaMallocPtr_ = nullptr;  // used for cudaMalloc
  gdr_t g_;
  gdr_mh_t memory_handle_;
  size_t size_ = 0;

  MappingDeviceMemory(size_t preAllocateSize = 1024 * GPU_PAGE_SIZE) {
    g_ = gdr_open();
    CHECK(g_);
    XMH_CUDA_CHECK(cudaMalloc(&cudaMallocPtr_, preAllocateSize));
    CHECK(cudaMallocPtr_);

    auto offset_dev_base_ptr =
        ((uint64_t)cudaMallocPtr_ + GPU_PAGE_SIZE - 1) & GPU_PAGE_MASK;
    size_ = (uint64_t)cudaMallocPtr_ + preAllocateSize - offset_dev_base_ptr;
    size_ = align_down(size_, GPU_PAGE_SIZE);
    dev_base_ptr_ = (void *)offset_dev_base_ptr;

    CHECK_EQ((uint64_t)dev_base_ptr_ & GPU_PAGE_OFFSET, 0);
    CHECK_GT(size_, 0);

    LOG(ERROR) << "Init MappingDeviceMemory: dev_base_ptr =" << dev_base_ptr_
               << "; offset_dev_base_ptr = " << offset_dev_base_ptr
               << "; size_ = " << size_;

    unsigned int flag = 1;
    ASSERTDRV(cuPointerSetAttribute(&flag, CU_POINTER_ATTRIBUTE_SYNC_MEMOPS,
                                    (CUdeviceptr)dev_base_ptr_));
    CHECK_EQ(
        0, gdr_pin_buffer(g_, (CUdeviceptr)dev_base_ptr_, size_, 0, 0, &memory_handle_));
    CHECK_EQ(gdr_map(g_, memory_handle_, &map_d_ptr, size_), 0);
    gdr_info_t info;
    CHECK_EQ(gdr_get_info(g_, memory_handle_, &info), 0);
    int off = info.va - (CUdeviceptr)dev_base_ptr_;
    map_base_ptr_ = (void *)((char *)map_d_ptr + off);
  }
};

#undef ASSERTDRV