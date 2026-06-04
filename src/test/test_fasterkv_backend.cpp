#include "storage/external/fasterkv/fasterkv_backend.h"

#include <filesystem>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

TEST(FasterKVBackendTest, InsertThenFetchReturnsFixedWidthValuesAndMisses) {
  recstore::storage::fasterkv::FasterKVBackend backend(128, 4);

  const long long keys[]         = {11, 22, 33};
  const std::vector<char> values = {
      'a',
      'a',
      'a',
      'a',
      'b',
      'b',
      'b',
      'b',
      'c',
      'c',
      'c',
      'c',
  };

  backend.Insert(3, keys, values.data());

  const long long query_keys[] = {33, 44, 11};
  std::vector<char> out(3 * 4, '?');
  size_t misses = 0;
  backend.Fetch(3, query_keys, out.data(), [&](size_t index) {
    EXPECT_EQ(1U, index);
    ++misses;
  });

  EXPECT_EQ(1U, misses);
  EXPECT_EQ(std::string("cccc"), std::string(out.data(), 4));
  EXPECT_EQ(std::string("????"), std::string(out.data() + 4, 4));
  EXPECT_EQ(std::string("aaaa"), std::string(out.data() + 8, 4));
}

TEST(FasterKVBackendTest, InsertCanOverwriteExistingFixedWidthValue) {
  recstore::storage::fasterkv::FasterKVBackend backend(128, 4);

  const long long key    = 7;
  const char old_value[] = {'o', 'l', 'd', '0'};
  const char new_value[] = {'n', 'e', 'w', '0'};
  backend.Insert(1, &key, old_value);
  backend.Insert(1, &key, new_value);

  std::vector<char> out(4, '?');
  size_t misses = 0;
  backend.Fetch(1, &key, out.data(), [&](size_t) { ++misses; });

  EXPECT_EQ(0U, misses);
  EXPECT_EQ(std::string("new0"), std::string(out.data(), out.size()));
}

TEST(FasterKVBackendTest, SsdDefaultHlogUsesMinimumMemoryWindow) {
  using recstore::storage::fasterkv::FasterKVStorage;
  constexpr uint64_t capacity                 = 10'000'000;
  constexpr size_t value_size                 = 128;
  constexpr uint64_t min_effective_hlog_bytes = 256ULL << 20;

  recstore::storage::fasterkv::FasterKVBackendOptions memory_options;
  memory_options.storage = FasterKVStorage::kMemory;
  EXPECT_GT(recstore::storage::fasterkv::ResolveHlogMemoryBytesForOptions(
                capacity, value_size, memory_options),
            min_effective_hlog_bytes);

  recstore::storage::fasterkv::FasterKVBackendOptions ssd_options;
  ssd_options.storage = FasterKVStorage::kSsd;
  EXPECT_EQ(recstore::storage::fasterkv::ResolveHlogMemoryBytesForOptions(
                capacity, value_size, ssd_options),
            min_effective_hlog_bytes);
}

TEST(FasterKVBackendTest, SsdBackendStoresValuesThroughFileSystemDisk) {
  recstore::storage::fasterkv::FasterKVBackendOptions options;
  options.storage  = recstore::storage::fasterkv::FasterKVStorage::kSsd;
  options.log_path = "/tmp/test_fasterkv_backend_ssd_" +
                     std::to_string(static_cast<long long>(getpid()));
  options.hlog_memory_bytes = 1ULL << 30;
  std::filesystem::remove_all(options.log_path);

  recstore::storage::fasterkv::FasterKVBackend backend(128, 4, options);

  const long long keys[]         = {101, 202};
  const std::vector<char> values = {'s', 's', 'd', '1', 's', 's', 'd', '2'};
  backend.Insert(2, keys, values.data());

  const long long query_keys[] = {202, 303, 101};
  std::vector<char> out(3 * 4, '?');
  size_t misses = 0;
  backend.Fetch(3, query_keys, out.data(), [&](size_t index) {
    EXPECT_EQ(1U, index);
    ++misses;
  });

  EXPECT_EQ(1U, misses);
  EXPECT_EQ(std::string("ssd2"), std::string(out.data(), 4));
  EXPECT_EQ(std::string("????"), std::string(out.data() + 4, 4));
  EXPECT_EQ(std::string("ssd1"), std::string(out.data() + 8, 4));

  std::filesystem::remove_all(options.log_path);
}

TEST(FasterKVBackendTest, RejectsInvalidMutableFraction) {
  recstore::storage::fasterkv::FasterKVBackendOptions options;
  options.mutable_fraction = 1.5;
  EXPECT_THROW(recstore::storage::fasterkv::FasterKVBackend(128, 4, options),
               std::invalid_argument);

  options.mutable_fraction = -0.1;
  EXPECT_THROW(recstore::storage::fasterkv::FasterKVBackend(128, 4, options),
               std::invalid_argument);
}
