#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/bind_core.h"
#include "base/init.h"
#include "base/log.h"
#include "storage/external/fasterkv/fasterkv_backend.h"
#include "storage/external/hps/hps_recstore.h"
#include "storage/external/hps/raw_rocksdb.h"
#include "third_party/HugeCTR/HugeCTR/include/hps/hash_map_backend.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/hier_parameter_server_base.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/inference_utils.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/rocksdb_backend.hpp"

DEFINE_string(backend,
              "hps_hash_map",
              "hps_hash_map|hps_rocksdb|raw_rocksdb|raw_rocksdb_memenv|"
              "fasterkv|recstore|hps_native_tiered");
DEFINE_string(path, "", "RecStore data path");
DEFINE_string(fasterkv_storage,
              "memory",
              "FasterKV storage backend: memory|ssd");
DEFINE_string(fasterkv_log_path,
              "",
              "FasterKV SSD log directory; defaults to path/fasterkv-log");
DEFINE_int64(fasterkv_hlog_memory_bytes,
             0,
             "FasterKV hybrid log memory bytes; 0 uses backend default");
DEFINE_double(fasterkv_mutable_fraction,
              0.0,
              "FasterKV hybrid log mutable fraction; 0 uses backend default");
DEFINE_int64(fasterkv_read_cache_bytes,
             0,
             "FasterKV read cache bytes; 0 disables read cache");
DEFINE_string(index_type, "DRAM_EXTENDIBLE_HASH", "RecStore index.type");
DEFINE_string(value_store_type, "DRAM_VALUE_STORE", "RecStore value.type");
DEFINE_string(dram_allocator, "PERSIST_LOOP_SLAB", "RecStore DRAM allocator");
DEFINE_int64(dram_capacity_bytes, 0, "override RecStore DRAM capacity bytes");
DEFINE_string(ssd_io_backend, "IOURING", "RecStore SSD IO backend");
DEFINE_string(ssd_value_file, "", "RecStore SSD value file");
DEFINE_int32(ssd_queue_depth, 512, "RecStore SSD IO queue depth");
DEFINE_int64(ssd_capacity_bytes, 0, "override RecStore SSD capacity bytes");
DEFINE_double(tiered_high_watermark_ratio,
              0.0,
              "override RecStore Tiered high watermark ratio; 0 uses default");
DEFINE_int64(record_count, 1000000, "record count");
DEFINE_int32(value_size, 512, "value size bytes");
DEFINE_int32(batch_size, 1024, "keys per HPS fetch/insert call");
DEFINE_int32(thread_num, 16, "worker thread count");
DEFINE_int32(load_thread_num, 0, "load thread count; 0 uses thread_num");
DEFINE_int32(hps_rocksdb_thread_num,
             1,
             "RocksDB internal thread count; 0 uses thread_num");
DEFINE_double(hps_native_dram_fraction,
              1.0,
              "HPS native tiered volatile DB capacity fraction in [0, 1]");
DEFINE_bool(
    hps_native_cache_missed_embeddings,
    false,
    "Pass through to HPS VolatileDatabaseParams.cache_missed_embeddings");
DEFINE_string(hps_native_overflow_policy,
              "evict_random",
              "HPS native volatile DB overflow policy: "
              "evict_random|evict_least_used|evict_oldest");
DEFINE_double(
    hps_native_overflow_resolution_target,
    0.8,
    "Pass through to HPS VolatileDatabaseParams.overflow_resolution_target");
DEFINE_int32(running_seconds, 5, "transaction runtime seconds");
DEFINE_string(distribution, "uniform", "uniform|zipfian");
DEFINE_double(zipfian_alpha, 0.9, "Zipfian alpha");
DEFINE_string(mode, "fetch", "fetch|insert|mixed|fetch_insert");
DEFINE_int32(read_ratio, 100, "read percentage for mixed mode");
DEFINE_string(table_name, "hps_recstore_bench_table", "HPS table name");

namespace {

struct PhaseStats {
  uint64_t batches = 0;
  uint64_t key_ops = 0;
  uint64_t misses  = 0;
};

class FastRandom {
public:
  explicit FastRandom(uint64_t seed)
      : state_(seed ? seed : 0x9e3779b97f4a7c15ULL) {}

  uint64_t Next() {
    uint64_t x = state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    state_ = x;
    return x * 2685821657736338717ULL;
  }

  double Uniform01() { return (Next() >> 11) * (1.0 / 9007199254740992.0); }
  uint64_t Uniform(uint64_t n) { return n == 0 ? 0 : Next() % n; }

private:
  uint64_t state_;
};

class KeyGenerator {
public:
  KeyGenerator(std::string distribution,
               uint64_t record_count,
               double alpha,
               uint64_t seed)
      : distribution_(std::move(distribution)),
        record_count_(record_count),
        alpha_(alpha),
        rng_(seed) {
    if (record_count_ == 0) {
      throw std::invalid_argument("record_count must be positive");
    }
    if (distribution_ == "zipfian") {
      if (std::abs(alpha_ - 1.0) < 1e-9) {
        log_n_ = std::log(static_cast<double>(record_count_));
      } else {
        pow_n_ = std::pow(static_cast<double>(record_count_), 1.0 - alpha_);
      }
    } else if (distribution_ != "uniform") {
      throw std::invalid_argument("distribution must be uniform or zipfian");
    }
  }

  uint64_t NextKey() {
    if (distribution_ == "uniform") {
      return rng_.Uniform(record_count_) + 1;
    }
    return NextZipfian() + 1;
  }

  uint64_t NextUint(uint64_t n) { return rng_.Uniform(n); }

private:
  uint64_t NextZipfian() {
    const double u = std::max(rng_.Uniform01(), 1e-12);
    double rank    = 1.0;
    if (std::abs(alpha_ - 1.0) < 1e-9) {
      rank = std::exp(u * log_n_);
    } else {
      rank = std::pow(1.0 + u * (pow_n_ - 1.0), 1.0 / (1.0 - alpha_));
    }
    uint64_t key = static_cast<uint64_t>(rank);
    if (key >= record_count_) {
      key = record_count_ - 1;
    }
    return key;
  }

  std::string distribution_;
  uint64_t record_count_;
  double alpha_;
  double pow_n_ = 1.0;
  double log_n_ = 0.0;
  FastRandom rng_;
};

using HpsBackend = HugeCTR::DatabaseBackendBase<long long>;

class BenchmarkBackend {
public:
  virtual ~BenchmarkBackend() = default;

  virtual void
  Insert(const std::string& table_name,
         size_t num_keys,
         const long long* keys,
         const char* values,
         uint32_t value_size,
         size_t stride) = 0;

  virtual void
  Fetch(const std::string& table_name,
        size_t num_keys,
        const long long* keys,
        char* values,
        size_t value_size,
        const std::function<void(size_t)>& on_miss) = 0;

  virtual void FinishLoad(const std::string& table_name) { (void)table_name; }

  virtual std::string ExtraResultFields() const { return ""; }
};

class HpsBenchmarkBackend : public BenchmarkBackend {
public:
  explicit HpsBenchmarkBackend(std::unique_ptr<HpsBackend> backend)
      : backend_(std::move(backend)) {}

  HpsBackend* raw() { return backend_.get(); }

  void Insert(const std::string& table_name,
              size_t num_keys,
              const long long* keys,
              const char* values,
              uint32_t value_size,
              size_t stride) override {
    backend_->insert(table_name, num_keys, keys, values, value_size, stride);
  }

  void Fetch(const std::string& table_name,
             size_t num_keys,
             const long long* keys,
             char* values,
             size_t value_size,
             const std::function<void(size_t)>& on_miss) override {
    backend_->fetch(
        table_name,
        num_keys,
        keys,
        values,
        value_size,
        on_miss,
        std::chrono::nanoseconds::zero());
  }

  std::string ExtraResultFields() const override {
    const auto* recstore =
        dynamic_cast<const recstore::storage::HpsRecStoreBackend<long long>*>(
            backend_.get());
    return recstore ? recstore->extra_result_fields() : "";
  }

private:
  std::unique_ptr<HpsBackend> backend_;
};

class RawRocksDBBenchmarkBackend : public BenchmarkBackend {
public:
  RawRocksDBBenchmarkBackend(
      const std::string& path, size_t value_size, bool use_mem_env)
      : backend_(path, value_size, use_mem_env) {}

  void Insert(const std::string& table_name,
              size_t num_keys,
              const long long* keys,
              const char* values,
              uint32_t value_size,
              size_t stride) override {
    (void)table_name;
    if (value_size != stride) {
      throw std::invalid_argument("raw_rocksdb requires value_size == stride");
    }
    backend_.Insert(num_keys, keys, values);
  }

  void Fetch(const std::string& table_name,
             size_t num_keys,
             const long long* keys,
             char* values,
             size_t value_size,
             const std::function<void(size_t)>& on_miss) override {
    (void)table_name;
    (void)value_size;
    backend_.Fetch(num_keys, keys, values, on_miss);
  }

private:
  recstore::storage::RawRocksDBBackend backend_;
};

class FasterKVBenchmarkBackend : public BenchmarkBackend {
public:
  FasterKVBenchmarkBackend(
      uint64_t capacity,
      size_t value_size,
      const recstore::storage::fasterkv::FasterKVBackendOptions& options)
      : backend_(capacity, value_size, options) {}

  void Insert(const std::string& table_name,
              size_t num_keys,
              const long long* keys,
              const char* values,
              uint32_t value_size,
              size_t stride) override {
    (void)table_name;
    if (value_size != stride) {
      throw std::invalid_argument("fasterkv requires value_size == stride");
    }
    backend_.Insert(num_keys, keys, values);
  }

  void Fetch(const std::string& table_name,
             size_t num_keys,
             const long long* keys,
             char* values,
             size_t value_size,
             const std::function<void(size_t)>& on_miss) override {
    (void)table_name;
    (void)value_size;
    backend_.Fetch(num_keys, keys, values, on_miss);
  }

private:
  recstore::storage::fasterkv::FasterKVBackend backend_;
};

HugeCTR::DatabaseOverflowPolicy_t HpsNativeOverflowPolicy() {
  if (FLAGS_hps_native_overflow_policy == "evict_random") {
    return HugeCTR::DatabaseOverflowPolicy_t::EvictRandom;
  }
  if (FLAGS_hps_native_overflow_policy == "evict_least_used") {
    return HugeCTR::DatabaseOverflowPolicy_t::EvictLeastUsed;
  }
  if (FLAGS_hps_native_overflow_policy == "evict_oldest") {
    return HugeCTR::DatabaseOverflowPolicy_t::EvictOldest;
  }
  throw std::invalid_argument("hps_native_overflow_policy must be "
                              "evict_random|evict_least_used|evict_oldest");
}

size_t HpsNativeOverflowMarginPerPartition() {
  if (FLAGS_hps_native_dram_fraction < 0.0 ||
      FLAGS_hps_native_dram_fraction > 1.0) {
    throw std::invalid_argument("hps_native_dram_fraction must be in [0, 1]");
  }
  const size_t partitions = static_cast<size_t>(std::max(1, FLAGS_thread_num));
  const double target_records =
      static_cast<double>(FLAGS_record_count) * FLAGS_hps_native_dram_fraction;
  if (target_records <= 0.0) {
    return 1;
  }
  return std::max<size_t>(
      1,
      static_cast<size_t>(
          std::ceil(target_records / static_cast<double>(partitions))));
}

class HpsNativeTieredBenchmarkBackend : public BenchmarkBackend {
public:
  HpsNativeTieredBenchmarkBackend(
      const std::string& path, size_t value_size, size_t max_batch_size)
      : path_(path),
        sparse_model_path_(path_ + "/hps_native_sparse"),
        value_size_(value_size),
        embedding_vec_size_(value_size / sizeof(float)),
        max_batch_size_(max_batch_size) {
    if (value_size_ == 0 || value_size_ % sizeof(float) != 0) {
      throw std::invalid_argument(
          "hps_native_tiered requires float-aligned value_size");
    }
    std::filesystem::create_directories(sparse_model_path_);
    key_stream_.open(sparse_model_path_ + "/key",
                     std::ios::binary | std::ios::out | std::ios::trunc);
    vec_stream_.open(sparse_model_path_ + "/emb_vector",
                     std::ios::binary | std::ios::out | std::ios::trunc);
    if (!key_stream_.is_open() || !vec_stream_.is_open()) {
      throw std::runtime_error("failed to create HPS native sparse files");
    }
  }

  void Insert(const std::string& table_name,
              size_t num_keys,
              const long long* keys,
              const char* values,
              uint32_t value_size,
              size_t stride) override {
    (void)table_name;
    if (parameter_server_) {
      throw std::runtime_error(
          "hps_native_tiered benchmark does not support inserts after load");
    }
    if (value_size != value_size_ || stride < value_size_) {
      throw std::invalid_argument(
          "hps_native_tiered insert requires fixed-size rows");
    }
    std::lock_guard<std::mutex> guard(load_mu_);
    key_stream_.write(
        reinterpret_cast<const char*>(keys),
        static_cast<std::streamsize>(num_keys * sizeof(long long)));
    for (size_t i = 0; i < num_keys; ++i) {
      vec_stream_.write(
          values + i * stride, static_cast<std::streamsize>(value_size_));
    }
    loaded_records_ += num_keys;
  }

  void FinishLoad(const std::string& table_name) override {
    (void)table_name;
    if (parameter_server_) {
      return;
    }
    key_stream_.close();
    vec_stream_.close();

    HugeCTR::VolatileDatabaseParams volatile_db;
    volatile_db.type = HugeCTR::DatabaseType_t::ParallelHashMap;
    volatile_db.num_partitions =
        static_cast<size_t>(std::max(1, FLAGS_thread_num));
    volatile_db.max_batch_size  = max_batch_size_;
    volatile_db.overflow_margin = HpsNativeOverflowMarginPerPartition();
    volatile_db.overflow_policy = HpsNativeOverflowPolicy();
    volatile_db.overflow_resolution_target =
        FLAGS_hps_native_overflow_resolution_target;
    volatile_db.initial_cache_rate = FLAGS_hps_native_dram_fraction;
    volatile_db.cache_missed_embeddings =
        FLAGS_hps_native_cache_missed_embeddings;
    volatile_db.update_filters.clear();

    HugeCTR::PersistentDatabaseParams persistent_db;
    persistent_db.type           = HugeCTR::DatabaseType_t::RocksDB;
    persistent_db.path           = path_ + "/rocksdb";
    persistent_db.num_threads    = static_cast<size_t>(std::max(
        1,
        FLAGS_hps_rocksdb_thread_num > 0 ? FLAGS_hps_rocksdb_thread_num
                                            : FLAGS_thread_num));
    persistent_db.max_batch_size = max_batch_size_;
    persistent_db.update_filters.clear();

    HugeCTR::UpdateSourceParams update_source;
    update_source.type = HugeCTR::UpdateSourceType_t::Null;

    HugeCTR::InferenceParams params(
        kModelName,
        max_batch_size_,
        0.9f,
        "",
        std::vector<std::string>{sparse_model_path_},
        0,
        false,
        0.0f,
        true,
        false,
        1.0f,
        true,
        true,
        1,
        1,
        std::max(1, FLAGS_thread_num),
        0.0f,
        std::vector<int>{0},
        std::vector<float>{0.0f},
        volatile_db,
        persistent_db,
        update_source,
        1,
        0.0f,
        0.0f,
        std::vector<size_t>{max_batch_size_},
        std::vector<size_t>{embedding_vec_size_},
        std::vector<std::string>{kTableName},
        "",
        1,
        1,
        "",
        false,
        HugeCTR::EmbeddingCacheType_t::Dynamic,
        true,
        false,
        true,
        false,
        false,
        false);

    HugeCTR::parameter_server_config ps_config(
        std::map<std::string, std::vector<std::string>>{
            {kModelName, std::vector<std::string>{kTableName}}},
        std::map<std::string, std::vector<size_t>>{
            {kModelName, std::vector<size_t>{embedding_vec_size_}}},
        std::map<std::string, std::vector<size_t>>{
            {kModelName, std::vector<size_t>{max_batch_size_}}},
        std::vector<HugeCTR::InferenceParams>{params},
        volatile_db,
        persistent_db,
        update_source);
    parameter_server_ = HugeCTR::HierParameterServerBase::create(ps_config);
  }

  void Fetch(const std::string& table_name,
             size_t num_keys,
             const long long* keys,
             char* values,
             size_t value_size,
             const std::function<void(size_t)>& on_miss) override {
    (void)table_name;
    if (!parameter_server_) {
      throw std::runtime_error("hps_native_tiered fetch before FinishLoad");
    }
    if (value_size != value_size_) {
      throw std::invalid_argument(
          "hps_native_tiered fetch value_size mismatch");
    }
    thread_local std::vector<float> scratch;
    scratch.resize(num_keys * embedding_vec_size_);
    parameter_server_->lookup(keys, num_keys, scratch.data(), kModelName, 0);
    std::memcpy(values, scratch.data(), num_keys * value_size_);
    for (size_t i = 0; i < num_keys; ++i) {
      if (static_cast<uint64_t>(keys[i]) == 0 ||
          static_cast<uint64_t>(keys[i]) >
              static_cast<uint64_t>(FLAGS_record_count)) {
        on_miss(i);
      }
    }
  }

  std::string ExtraResultFields() const override {
    std::ostringstream os;
    os << " hps_native_dram_fraction=" << FLAGS_hps_native_dram_fraction
       << " hps_native_volatile_overflow_margin="
       << HpsNativeOverflowMarginPerPartition()
       << " hps_native_cache_missed_embeddings="
       << (FLAGS_hps_native_cache_missed_embeddings ? "true" : "false")
       << " hps_native_loaded_records=" << loaded_records_;
    return os.str();
  }

private:
  static constexpr const char* kModelName = "recstore_hps_native_bench";
  static constexpr const char* kTableName = "table0";

  std::string path_;
  std::string sparse_model_path_;
  size_t value_size_;
  size_t embedding_vec_size_;
  size_t max_batch_size_;
  size_t loaded_records_ = 0;
  std::mutex load_mu_;
  std::ofstream key_stream_;
  std::ofstream vec_stream_;
  std::shared_ptr<HugeCTR::HierParameterServerBase> parameter_server_;
};

recstore::storage::fasterkv::FasterKVBackendOptions FasterKvOptions() {
  recstore::storage::fasterkv::FasterKVBackendOptions options;
  if (FLAGS_fasterkv_storage == "memory") {
    options.storage = recstore::storage::fasterkv::FasterKVStorage::kMemory;
  } else if (FLAGS_fasterkv_storage == "ssd") {
    options.storage = recstore::storage::fasterkv::FasterKVStorage::kSsd;
  } else {
    throw std::invalid_argument("fasterkv_storage must be memory or ssd");
  }
  options.log_path = FLAGS_fasterkv_log_path;
  if (options.storage == recstore::storage::fasterkv::FasterKVStorage::kSsd &&
      options.log_path.empty()) {
    if (FLAGS_path.empty()) {
      throw std::invalid_argument(
          "path or fasterkv_log_path is required for fasterkv_storage=ssd");
    }
    options.log_path = FLAGS_path + "/fasterkv-log";
  }
  if (FLAGS_fasterkv_hlog_memory_bytes > 0) {
    options.hlog_memory_bytes =
        static_cast<uint64_t>(FLAGS_fasterkv_hlog_memory_bytes);
  }
  if (FLAGS_fasterkv_mutable_fraction < 0.0 ||
      FLAGS_fasterkv_mutable_fraction > 1.0) {
    throw std::invalid_argument("fasterkv_mutable_fraction must be in [0, 1]");
  }
  options.mutable_fraction = FLAGS_fasterkv_mutable_fraction;
  if (FLAGS_fasterkv_read_cache_bytes > 0) {
    options.read_cache_bytes =
        static_cast<uint64_t>(FLAGS_fasterkv_read_cache_bytes);
  }
  return options;
}

const std::string& EffectiveTableName() {
  static const std::string kRocksDbDefaultTable = "default";
  return FLAGS_backend == "hps_rocksdb"
           ? kRocksDbDefaultTable
           : FLAGS_table_name;
}

std::unique_ptr<HpsBackend> CreateBackend() {
  if (FLAGS_backend == "hps_hash_map") {
    HugeCTR::HashMapBackendParams params;
    params.max_batch_size  = static_cast<size_t>(FLAGS_batch_size);
    params.num_partitions  = static_cast<size_t>(std::max(1, FLAGS_thread_num));
    params.overflow_margin = std::numeric_limits<size_t>::max();
    return std::make_unique<HugeCTR::HashMapBackend<long long>>(params);
  }
  if (FLAGS_backend == "hps_rocksdb") {
    HugeCTR::RocksDBBackendParams params;
    params.path           = FLAGS_path;
    params.max_batch_size = static_cast<size_t>(FLAGS_batch_size);
    const int rocksdb_threads =
        FLAGS_hps_rocksdb_thread_num > 0
            ? FLAGS_hps_rocksdb_thread_num
            : FLAGS_thread_num;
    params.num_threads = static_cast<size_t>(std::max(1, rocksdb_threads));
    return std::make_unique<HugeCTR::RocksDBBackend<long long>>(params);
  }
  if (FLAGS_backend == "recstore") {
    recstore::storage::HpsRecStoreBackendParams params;
    params.path           = FLAGS_path;
    params.capacity       = static_cast<uint64_t>(FLAGS_record_count);
    params.value_size     = static_cast<uint32_t>(FLAGS_value_size);
    params.max_batch_size = static_cast<size_t>(FLAGS_batch_size);
    params.num_partitions = static_cast<size_t>(std::max(1, FLAGS_thread_num));
    params.index_type     = FLAGS_index_type;
    params.value_store_type = FLAGS_value_store_type;
    params.dram_allocator   = FLAGS_dram_allocator;
    params.dram_capacity_bytes =
        FLAGS_dram_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_dram_capacity_bytes)
            : 0;
    params.ssd_capacity_bytes =
        FLAGS_ssd_capacity_bytes > 0
            ? static_cast<uint64_t>(FLAGS_ssd_capacity_bytes)
            : 0;
    params.tiered_high_watermark_ratio = FLAGS_tiered_high_watermark_ratio;
    params.ssd_io_backend              = FLAGS_ssd_io_backend;
    params.ssd_value_file              = FLAGS_ssd_value_file;
    params.ssd_queue_depth             = FLAGS_ssd_queue_depth;
    params.num_threads                 = FLAGS_thread_num;
    return std::make_unique<recstore::storage::HpsRecStoreBackend<long long>>(
        params);
  }
  throw std::invalid_argument("unsupported backend: " + FLAGS_backend);
}

std::unique_ptr<BenchmarkBackend> CreateBenchmarkBackend() {
  if (FLAGS_backend == "raw_rocksdb" || FLAGS_backend == "raw_rocksdb_memenv") {
    return std::make_unique<RawRocksDBBenchmarkBackend>(
        FLAGS_path,
        static_cast<size_t>(FLAGS_value_size),
        FLAGS_backend == "raw_rocksdb_memenv");
  }
  if (FLAGS_backend == "hps_native_tiered") {
    return std::make_unique<HpsNativeTieredBenchmarkBackend>(
        FLAGS_path,
        static_cast<size_t>(FLAGS_value_size),
        static_cast<size_t>(FLAGS_batch_size));
  }
  if (FLAGS_backend == "fasterkv") {
    return std::make_unique<FasterKVBenchmarkBackend>(
        static_cast<uint64_t>(FLAGS_record_count),
        static_cast<size_t>(FLAGS_value_size),
        FasterKvOptions());
  }
  return std::make_unique<HpsBenchmarkBackend>(CreateBackend());
}

std::vector<char> MakeValues(size_t rows, size_t value_size, int seed) {
  std::vector<char> values(rows * value_size);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<char>('a' + ((i + seed) % 26));
  }
  return values;
}

PhaseStats LoadRecords(BenchmarkBackend* backend, int load_threads) {
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(load_threads);
  const uint64_t record_count = static_cast<uint64_t>(FLAGS_record_count);
  const uint64_t per_thread =
      (record_count + static_cast<uint64_t>(load_threads) - 1) /
      static_cast<uint64_t>(load_threads);

  for (int tid = 0; tid < load_threads; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      std::vector<long long> keys;
      keys.reserve(static_cast<size_t>(FLAGS_batch_size));
      std::vector<char> values = MakeValues(
          static_cast<size_t>(FLAGS_batch_size), FLAGS_value_size, tid);
      const uint64_t begin = static_cast<uint64_t>(tid) * per_thread + 1;
      const uint64_t end   = std::min(record_count + 1, begin + per_thread);
      PhaseStats local;
      for (uint64_t key = begin; key < end;) {
        keys.clear();
        while (key < end &&
               keys.size() < static_cast<size_t>(FLAGS_batch_size)) {
          keys.push_back(static_cast<long long>(key++));
        }
        backend->Insert(
            EffectiveTableName(),
            keys.size(),
            keys.data(),
            values.data(),
            static_cast<uint32_t>(FLAGS_value_size),
            static_cast<size_t>(FLAGS_value_size));
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
  }
  return total;
}

void PrepareBackendForLoad(BenchmarkBackend* backend) {
  if (FLAGS_backend != "hps_rocksdb") {
    return;
  }

  // RocksDBBackend lazily creates non-default column families on insert. Use
  // the default column family in this benchmark to avoid that unstable path.
  const long long key = 1;
  std::vector<char> value =
      MakeValues(1, static_cast<size_t>(FLAGS_value_size), 0);
  backend->Insert(
      EffectiveTableName(),
      1,
      &key,
      value.data(),
      static_cast<uint32_t>(FLAGS_value_size),
      static_cast<size_t>(FLAGS_value_size));
}

PhaseStats RunTransactions(BenchmarkBackend* backend) {
  const bool fetch_only   = FLAGS_mode == "fetch";
  const bool insert_only  = FLAGS_mode == "insert";
  const bool mixed        = FLAGS_mode == "mixed";
  const bool fetch_insert = FLAGS_mode == "fetch_insert";
  if (!fetch_only && !insert_only && !mixed && !fetch_insert) {
    throw std::invalid_argument("mode must be fetch|insert|mixed|fetch_insert");
  }

  std::atomic<bool> start{false};
  std::atomic<bool> stop{false};
  std::vector<std::thread> threads;
  std::vector<PhaseStats> stats(FLAGS_thread_num);

  for (int tid = 0; tid < FLAGS_thread_num; ++tid) {
    threads.emplace_back([&, tid]() {
      base::auto_bind_core();
      KeyGenerator key_gen(
          FLAGS_distribution,
          static_cast<uint64_t>(FLAGS_record_count),
          FLAGS_zipfian_alpha,
          0x9e3779b97f4a7c15ULL + static_cast<uint64_t>(tid));
      std::vector<long long> keys(static_cast<size_t>(FLAGS_batch_size));
      std::vector<char> values = MakeValues(
          static_cast<size_t>(FLAGS_batch_size), FLAGS_value_size, tid);
      std::vector<char> out(static_cast<size_t>(FLAGS_batch_size) *
                            static_cast<size_t>(FLAGS_value_size));
      PhaseStats local;
      while (!start.load(std::memory_order_acquire)) {
      }
      while (!stop.load(std::memory_order_relaxed)) {
        for (auto& key : keys) {
          key = static_cast<long long>(key_gen.NextKey());
        }
        const bool do_fetch =
            fetch_only || fetch_insert ||
            (mixed &&
             static_cast<int>(key_gen.NextUint(100)) < FLAGS_read_ratio);
        if (do_fetch) {
          size_t misses = 0;
          backend->Fetch(
              EffectiveTableName(),
              keys.size(),
              keys.data(),
              out.data(),
              static_cast<size_t>(FLAGS_value_size),
              [&](size_t) { ++misses; });
          local.misses += misses;
        }
        if (insert_only || fetch_insert || (mixed && !do_fetch)) {
          backend->Insert(
              EffectiveTableName(),
              keys.size(),
              keys.data(),
              values.data(),
              static_cast<uint32_t>(FLAGS_value_size),
              static_cast<size_t>(FLAGS_value_size));
        }
        ++local.batches;
        local.key_ops += keys.size();
      }
      stats[tid] = local;
    });
  }

  start.store(true, std::memory_order_release);
  sleep(FLAGS_running_seconds);
  stop.store(true, std::memory_order_relaxed);
  for (auto& thread : threads) {
    thread.join();
  }

  PhaseStats total;
  for (const auto& each : stats) {
    total.batches += each.batches;
    total.key_ops += each.key_ops;
    total.misses += each.misses;
  }
  return total;
}

double SecondsSince(std::chrono::steady_clock::time_point start,
                    std::chrono::steady_clock::time_point end) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(end - start)
      .count();
}

void PrintResult(const char* phase,
                 const PhaseStats& stats,
                 double seconds,
                 const BenchmarkBackend* backend) {
  const double batch_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.batches) / seconds : 0.0;
  const double key_ops_sec =
      seconds > 0.0 ? static_cast<double>(stats.key_ops) / seconds : 0.0;
  const bool hps_native_tiered = FLAGS_backend == "hps_native_tiered";
  const char* index_type =
      hps_native_tiered ? "HPS_PARALLEL_HASH_MAP" : FLAGS_index_type.c_str();
  const char* value_store_type =
      hps_native_tiered ? "HPS_ROCKSDB" : FLAGS_value_store_type.c_str();
  std::printf(
      "BACKEND_BENCHMARK_RESULT phase=%s backend=%s index_type=%s "
      "value_store_type=%s mode=%s distribution=%s zipfian_alpha=%.6f "
      "threads=%d batch_size=%d records=%ld runtime_s=%.6f batches=%lu "
      "key_ops=%lu misses=%lu throughput_batches_sec=%.6f "
      "throughput_keys_sec=%.6f%s\n",
      phase,
      FLAGS_backend.c_str(),
      index_type,
      value_store_type,
      FLAGS_mode.c_str(),
      FLAGS_distribution.c_str(),
      FLAGS_zipfian_alpha,
      FLAGS_thread_num,
      FLAGS_batch_size,
      FLAGS_record_count,
      seconds,
      stats.batches,
      stats.key_ops,
      stats.misses,
      batch_ops_sec,
      key_ops_sec,
      backend == nullptr ? "" : backend->ExtraResultFields().c_str());
}

} // namespace

int main(int argc, char** argv) {
  base::Init(&argc, &argv);
  CHECK_GT(FLAGS_record_count, 0);
  CHECK_GT(FLAGS_value_size, 0);
  CHECK_GT(FLAGS_batch_size, 0);
  CHECK_GT(FLAGS_thread_num, 0);
  CHECK_GT(FLAGS_running_seconds, 0);
  if (FLAGS_backend == "recstore" || FLAGS_backend == "hps_rocksdb" ||
      FLAGS_backend == "hps_native_tiered" || FLAGS_backend == "raw_rocksdb" ||
      FLAGS_backend == "raw_rocksdb_memenv") {
    CHECK(!FLAGS_path.empty())
        << "--path is required for " << FLAGS_backend << " backend";
  }

  const int load_threads =
      FLAGS_load_thread_num > 0 ? FLAGS_load_thread_num : FLAGS_thread_num;
  auto backend = CreateBenchmarkBackend();
  PrepareBackendForLoad(backend.get());

  const auto load_begin = std::chrono::steady_clock::now();
  const PhaseStats load = LoadRecords(backend.get(), load_threads);
  backend->FinishLoad(EffectiveTableName());
  const auto load_end = std::chrono::steady_clock::now();
  PrintResult("load", load, SecondsSince(load_begin, load_end), backend.get());

  const auto run_begin = std::chrono::steady_clock::now();
  const PhaseStats run = RunTransactions(backend.get());
  const auto run_end   = std::chrono::steady_clock::now();
  PrintResult("run", run, SecondsSince(run_begin, run_end), backend.get());
  return 0;
}
