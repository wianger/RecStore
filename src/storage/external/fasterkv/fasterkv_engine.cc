#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "base/factory.h"
#include "storage/external/fasterkv/fasterkv_backend.h"
#include "storage/kv_engine/base_kv.h"

namespace {

size_t ConfigValueSize(const BaseKVConfig& config) {
  const auto& j = config.json_config_;
  if (j.contains("value_size")) {
    return j.at("value_size").get<size_t>();
  }
  if (j.contains("value")) {
    return j.at("value").value("default_value_size_hint", 0);
  }
  return 0;
}

recstore::storage::fasterkv::FasterKVBackendOptions
ConfigBackendOptions(const BaseKVConfig& config) {
  recstore::storage::fasterkv::FasterKVBackendOptions options;
  const auto& j = config.json_config_;
  if (!j.contains("fasterkv")) {
    return options;
  }
  const auto& fkv           = j.at("fasterkv");
  const std::string storage = fkv.value("storage", "memory");
  if (storage == "memory") {
    options.storage = recstore::storage::fasterkv::FasterKVStorage::kMemory;
  } else if (storage == "ssd") {
    options.storage = recstore::storage::fasterkv::FasterKVStorage::kSsd;
  } else {
    throw std::invalid_argument(
        "KVEngineFasterKV fasterkv.storage must be memory or ssd");
  }
  if (fkv.contains("log_path")) {
    options.log_path = fkv.at("log_path").get<std::string>();
  } else if (options.storage ==
                 recstore::storage::fasterkv::FasterKVStorage::kSsd &&
             j.contains("path") && !j.at("path").get<std::string>().empty()) {
    options.log_path = j.at("path").get<std::string>() + "/fasterkv-log";
  }
  if (fkv.contains("hlog_memory_bytes")) {
    options.hlog_memory_bytes = fkv.at("hlog_memory_bytes").get<uint64_t>();
  }
  if (fkv.contains("mutable_fraction")) {
    options.mutable_fraction = fkv.at("mutable_fraction").get<double>();
  }
  if (fkv.contains("read_cache_bytes")) {
    options.read_cache_bytes = fkv.at("read_cache_bytes").get<uint64_t>();
  }
  if (options.mutable_fraction < 0.0 || options.mutable_fraction > 1.0) {
    throw std::invalid_argument(
        "KVEngineFasterKV fasterkv.mutable_fraction must be in [0, 1]");
  }
  return options;
}

void ValidateFloatAligned(size_t value_size, const char* operation) {
  if (value_size == 0 || value_size % sizeof(float) != 0) {
    throw std::invalid_argument(
        std::string(operation) +
        " requires a non-zero float-aligned value_size");
  }
}

std::vector<long long> ConvertKeys(base::ConstArray<uint64_t> keys) {
  std::vector<long long> out(keys.Size());
  for (int i = 0; i < keys.Size(); ++i) {
    out[i] = static_cast<long long>(keys[i]);
  }
  return out;
}

} // namespace

class KVEngineFasterKV : public BaseKV {
public:
  explicit KVEngineFasterKV(const BaseKVConfig& config)
      : BaseKV(config),
        value_size_(ConfigValueSize(config)),
        backend_(config.json_config_.at("capacity").get<uint64_t>(),
                 value_size_,
                 ConfigBackendOptions(config)) {
    if (value_size_ == 0) {
      throw std::invalid_argument("KVEngineFasterKV requires value_size");
    }
  }

  void Get(const uint64_t key, std::string& value, unsigned tid) override {
    (void)tid;
    const long long hps_key = static_cast<long long>(key);
    value.assign(value_size_, '\0');
    bool missed = false;
    backend_.Fetch(1, &hps_key, value.data(), [&](size_t) { missed = true; });
    if (missed) {
      value.clear();
    }
  }

  bool Exists(const uint64_t key, unsigned tid) override {
    (void)tid;
    const long long hps_key = static_cast<long long>(key);
    std::string value(value_size_, '\0');
    bool missed = false;
    backend_.Fetch(1, &hps_key, value.data(), [&](size_t) { missed = true; });
    return !missed;
  }

  void Put(const uint64_t key,
           const std::string_view& value,
           unsigned tid) override {
    (void)tid;
    if (value.size() != value_size_) {
      throw std::invalid_argument("KVEngineFasterKV requires fixed-size Put");
    }
    const long long hps_key = static_cast<long long>(key);
    backend_.Insert(1, &hps_key, value.data());
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "KVEngineFasterKV::BatchPut");
    if (values == nullptr || keys.Size() != static_cast<int>(values->size())) {
      throw std::invalid_argument("KVEngineFasterKV::BatchPut size mismatch");
    }

    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    for (int i = 0; i < keys.Size(); ++i) {
      const auto& row = (*values)[i];
      if (row.Size() != floats_per_row) {
        throw std::invalid_argument(
            "KVEngineFasterKV::BatchPut row size mismatch");
      }
      std::memcpy(flat.data() + static_cast<size_t>(i) * value_size_,
                  row.Data(),
                  value_size_);
    }
    std::vector<long long> hps_keys = ConvertKeys(keys);
    backend_.Insert(hps_keys.size(), hps_keys.data(), flat.data());
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>>* values,
                unsigned tid) override {
    (void)tid;
    ValidateFloatAligned(value_size_, "KVEngineFasterKV::BatchGet");
    values->resize(keys.Size());
    thread_local std::vector<std::vector<float>> buffers;
    buffers.assign(keys.Size(), {});

    std::vector<long long> hps_keys = ConvertKeys(keys);
    std::vector<char> flat(static_cast<size_t>(keys.Size()) * value_size_);
    std::vector<uint8_t> misses(keys.Size(), 0);
    backend_.Fetch(
        hps_keys.size(), hps_keys.data(), flat.data(), [&](size_t index) {
          misses[index] = 1;
        });

    const int floats_per_row = static_cast<int>(value_size_ / sizeof(float));
    for (int i = 0; i < keys.Size(); ++i) {
      if (misses[i]) {
        (*values)[i] = base::ConstArray<float>();
        continue;
      }
      buffers[i].resize(floats_per_row);
      std::memcpy(buffers[i].data(),
                  flat.data() + static_cast<size_t>(i) * value_size_,
                  value_size_);
      (*values)[i] =
          base::ConstArray<float>(buffers[i].data(), buffers[i].size());
    }
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void* value) override {
    if (value == nullptr) {
      throw std::invalid_argument("KVEngineFasterKV::BulkLoad value is null");
    }
    std::vector<long long> hps_keys = ConvertKeys(keys);
    backend_.Insert(
        hps_keys.size(), hps_keys.data(), reinterpret_cast<const char*>(value));
  }

private:
  size_t value_size_;
  recstore::storage::fasterkv::FasterKVBackend backend_;
};

FACTORY_REGISTER(
    BaseKV, KVEngineFasterKV, KVEngineFasterKV, const BaseKVConfig&);
