#include <memory>
#include <string>
#include <assert.h>
#include <folly/GLog.h>
#include <folly/ProducerConsumerQueue.h>
#include "rmm/mr/device/cuda_memory_resource.hpp"
#include "rmm/mr/device/device_memory_resource.hpp"
#include "rmm/mr/device/managed_memory_resource.hpp"
#include "rmm/mr/device/pool_memory_resource.hpp"
#include "rmm/mr/device/binning_memory_resource.hpp"
#include "inference/timer.h"

#include "memory_pool.h"

namespace xmh {

PoolMemoryResource::PoolMemoryResource(std::size_t initial_pool_size,
                                       std::size_t max_pool_size,
                                       const std::string &type) {
  if (type == "device") {
    SourceMr_.reset(new rmm::mr::cuda_memory_resource());
  } else if (type == "managed") {
    SourceMr_.reset(new rmm::mr::managed_memory_resource());
  } else {
    assert(false);
    while (1) fprintf(stderr, "PoolMemoryResource type error");
  }

  initial_pool_size = xmh::MathUtil::round_up_to(initial_pool_size, 256);
  CHECK(initial_pool_size % 256 == 0);
  if (max_pool_size == -1) {
    pooled_mrs_.reset(new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        SourceMr_.get(), initial_pool_size));
  } else {
    max_pool_size = xmh::MathUtil::round_up_to(max_pool_size, 256);
    CHECK(max_pool_size % 256 == 0);
    pooled_mrs_.reset(new rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource>(
        SourceMr_.get(), initial_pool_size, max_pool_size));
  }
}

void *PoolMemoryResource::allocate(std::size_t bytes, cudaStream_t stream) {
  void *p = pooled_mrs_->allocate(bytes, stream);
  return p;
}

void PoolMemoryResource::deallocate(void *p, std::size_t bytes, cudaStream_t stream) {
  pooled_mrs_->deallocate(p, bytes, stream);
}

PoolMemoryResource::~PoolMemoryResource() {}

BinningMemoryResource::BinningMemoryResource(int8_t min_size_exponent,
                                             int8_t max_size_exponent,
                                             const std::string &type) {
  if (type == "device") {
    SourceMr_.reset(new rmm::mr::cuda_memory_resource());
  } else if (type == "managed") {
    SourceMr_.reset(new rmm::mr::managed_memory_resource());
  } else {
    assert(false);
    while (1) fprintf(stderr, "BinningMemoryResource type error");
  }
  binning_mrs_.reset(
      new rmm::mr::binning_memory_resource<rmm::mr::device_memory_resource>(
          SourceMr_.get(), min_size_exponent, max_size_exponent));
}

void *BinningMemoryResource::allocate(std::size_t bytes, cudaStream_t stream) {
  void *p = binning_mrs_->allocate(bytes, stream);
  return p;
}

void BinningMemoryResource::deallocate(void *p, std::size_t bytes, cudaStream_t stream) {
  binning_mrs_->deallocate(p, bytes, stream);
}

BinningMemoryResource::~BinningMemoryResource() {}

FixedLengthMemoryResource::FixedLengthMemoryResource(
    std::size_t initial_pool_count, const std::vector<int> &support_bytes,
    const std::string &type) {
  initial_pool_count_ = initial_pool_count;
  if (type == "device") {
    SourceMr_.reset(new rmm::mr::cuda_memory_resource());
  } else if (type == "managed") {
    SourceMr_.reset(new rmm::mr::managed_memory_resource());
  } else {
    assert(false);
    while (1) fprintf(stderr, "FixedLengthMemoryResource type error");
  }
  for (auto &each : support_bytes) {
    void *p = SourceMr_->allocate(initial_pool_count * each);
    allocate_start_addrs_.emplace_back(p, initial_pool_count * each);
    if (initial_pool_count == 0) {
      memory_pool_[each] = new folly::ProducerConsumerQueue<void *>(2);
    } else {
      memory_pool_[each] =
          new folly::ProducerConsumerQueue<void *>(100 * initial_pool_count);
    }
    for (int i = 0; i < initial_pool_count; i++) {
      CHECK(memory_pool_[each]->write((char *)p + i * each));
    }
  }
}

void *FixedLengthMemoryResource::allocate(std::size_t bytes) {
  void *q;
  if (memory_pool_[bytes]->isEmpty()) {
    void *p = SourceMr_->allocate(initial_pool_count_ * bytes);
    allocate_start_addrs_.emplace_back(p, initial_pool_count_ * bytes);
    for (int i = 0; i < initial_pool_count_; i++) {
      CHECK(memory_pool_[bytes]->write((char *)p + i * bytes));
    }
  }
  CHECK(memory_pool_[bytes]->read(q));
  return q;
}

void FixedLengthMemoryResource::deallocate(void *p, std::size_t bytes) {
  CHECK(memory_pool_[bytes]->write(p));
}

FixedLengthMemoryResource::~FixedLengthMemoryResource() {
  for (const auto &[key, value] : memory_pool_) {
    delete value;
  }

  for (auto &p : allocate_start_addrs_) {
    SourceMr_->deallocate(p.first, p.second);
  }
}

}  // namespace xmh