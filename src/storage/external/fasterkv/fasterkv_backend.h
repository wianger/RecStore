#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace recstore::storage::fasterkv {

enum class FasterKVStorage {
  kMemory,
  kSsd,
};

struct FasterKVBackendOptions {
  FasterKVStorage storage = FasterKVStorage::kMemory;
  std::string log_path;
  uint64_t hlog_memory_bytes = 0;
  double mutable_fraction    = 0.0;
  uint64_t read_cache_bytes  = 0;
};

class FasterKVBackend {
public:
  using MissCallback = std::function<void(size_t)>;

  FasterKVBackend(uint64_t capacity, size_t value_size);
  FasterKVBackend(uint64_t capacity,
                  size_t value_size,
                  const FasterKVBackendOptions& options);
  ~FasterKVBackend();

  FasterKVBackend(const FasterKVBackend&)            = delete;
  FasterKVBackend& operator=(const FasterKVBackend&) = delete;

  void Insert(size_t num_keys, const long long* keys, const char* values);

  void Fetch(size_t num_keys,
             const long long* keys,
             char* values,
             const MissCallback& on_miss);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace recstore::storage::fasterkv
