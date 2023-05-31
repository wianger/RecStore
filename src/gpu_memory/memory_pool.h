#pragma once

#include <memory>
#include <string>

namespace rmm {
namespace mr {
class device_memory_resource;

template <typename Upstream>
class pool_memory_resource;

template <typename Upstream>
class binning_memory_resource;
}  // namespace mr
}  // namespace rmm

namespace folly {
template <class T>
struct ProducerConsumerQueue;
}  // namespace folly

namespace xmh {
class PoolMemoryResource {
 public:
  PoolMemoryResource(std::size_t initial_pool_size, std::size_t max_pool_size,
                     const std::string &type);
  void *allocate(std::size_t bytes, cudaStream_t stream = 0);
  void deallocate(void *p, std::size_t bytes, cudaStream_t stream = 0);

  virtual ~PoolMemoryResource();

 private:
  std::unique_ptr<rmm::mr::device_memory_resource> SourceMr_;
  std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>>
      pooled_mrs_;
};

class BinningMemoryResource {
 public:
  BinningMemoryResource(int8_t min_size_exponent, int8_t max_size_exponent,
                        const std::string &type);

  void *allocate(std::size_t bytes, cudaStream_t stream = 0);
  void deallocate(void *p, std::size_t bytes, cudaStream_t stream = 0);

  virtual ~BinningMemoryResource();

 private:
  std::unique_ptr<rmm::mr::device_memory_resource> SourceMr_;
  std::unique_ptr<rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>>
      binning_mrs_;
};

// This is not thread safe
class FixedLengthMemoryResource {
 public:
  FixedLengthMemoryResource(std::size_t initial_pool_count,
                            const std::vector<int> &support_bytes,
                            const std::string &type);

  void *allocate(std::size_t bytes);
  void deallocate(void *p, std::size_t bytes);

  virtual ~FixedLengthMemoryResource();

  void *StartAddr() const {
    return allocate_start_addrs_[0].first;
  }

  void *EndAddr() const {
    return (char *)allocate_start_addrs_[0].first + allocate_start_addrs_[0].second;
  }

 private:
  std::unique_ptr<rmm::mr::device_memory_resource> SourceMr_;
  std::unordered_map<int, folly::ProducerConsumerQueue<void *> *> memory_pool_;
  int initial_pool_count_;
  std::vector<std::pair<void *, int>> allocate_start_addrs_;
};

}  // namespace xmh