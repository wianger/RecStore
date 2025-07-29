#pragma once

#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "../dram/extendible_hash.h"
#include "base/factory.h"
#include "base_kv.h"
#include "memory/persist_malloc.h"

class KVEngineHybrid : public BaseKV {
  static constexpr int kKVEngineValidFileSize = 123;

public:
  KVEngineHybrid(const BaseKVConfig &config)
      : BaseKV(config),
        valm(config.json_config_.at("path").get<std::string>() +
                        "/value",
                    1.2 * config.json_config_.at("capacity").get<size_t>() *
                        config.json_config_.at("value_size").get<size_t>())

  {
    value_size_ = config.json_config_.at("value_size").get<int>();

    // 初始化extendible hash表
    IndexConfig index_config;
    index_config.json_config_ = config.json_config_;
    Index *index = new ExtendibleHash(index_config);

    std::string path = config.json_config_.at("path").get<std::string>();

    // 初始化值存储区域
    uint64_t value_shm_size =
        config.json_config_.at("capacity").get<uint64_t>() *
        config.json_config_.at("value_size").get<uint64_t>();

    // if (!valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize)) {
    //   base::file_util::Delete(path + "/valid", false);
    //   CHECK(
    //       valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize));
    //   shm_malloc_.Initialize();
    // }
    // LOG(INFO) << "After init: [shm_malloc] " << shm_malloc_.GetInfo();
  }

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    // TODO:
    // 1. get <key, pointer> from index
    // 2. get value length from value header
    // 2. get value from pointer and length
    uint64_t pointer;
    index->Get(key, pointer, tid);
    value = valm.RetrieveValue(UnifiedPointer::FromRaw(pointer));

  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    // TODO: 
    // 1. malloc space for value
    //     -> value length in value header
    //     -> return the unified pointer to value
    // 2. insert <key, pointer> into index
    UnifiedPointer p = valm.WriteValue(value);
    index->Put(key, p.value(), tid);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<char>> *values,
                unsigned tid) override {
    values->clear();
    // std::shared_lock<std::shared_mutex> _(lock_);

    for (auto k : keys) {
      uint64_t pointer;
      index->Get(k, pointer, tid);
      std::string temp_values = valm.RetrieveValue(UnifiedPointer::FromRaw(pointer));
      values->emplace_back(temp_values.data(),temp_values.size());
    }
  }

  ~KVEngineHybrid() {
    std::cout << "exit KVEngineHybrid" << std::endl;
    if (index) {
      delete index;
      index = nullptr;
    }
  }

private:
  Index *index = nullptr;
  ValueManager valm;
  // std::shared_mutex lock_;

  uint64_t counter = 0;
  std::string dict_pool_name_;
  size_t dict_pool_size_;
  int value_size_;
  // base::PersistLoopShmMalloc shm_malloc_;
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid,
                 const BaseKVConfig &);