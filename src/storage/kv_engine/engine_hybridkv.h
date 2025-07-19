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
#ifdef XMH_SIMPLE_MALLOC
        shm_malloc_(config.json_config_.at("path").get<std::string>() +
                        "/value",
                    config.json_config_.at("capacity").get<size_t>() *
                        config.json_config_.at("value_size").get<size_t>(),
                    config.json_config_.at("value_size").get<size_t>())
#else
        shm_malloc_(config.json_config_.at("path").get<std::string>() +
                        "/value",
                    1.2 * config.json_config_.at("capacity").get<size_t>() *
                        config.json_config_.at("value_size").get<size_t>())
#endif
  {
    value_size_ = config.json_config_.at("value_size").get<int>();

    // 初始化extendible hash表
    Index *index = new ExtendibleHash();

    std::string path = config.json_config_.at("path").get<std::string>();

    // 初始化值存储区域
    uint64_t value_shm_size =
        config.json_config_.at("capacity").get<uint64_t>() *
        config.json_config_.at("value_size").get<uint64_t>();

    if (!valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize)) {
      base::file_util::Delete(path + "/valid", false);
      CHECK(
          valid_shm_file_.Initialize(path + "/valid", kKVEngineValidFileSize));
      shm_malloc_.Initialize();
    }
    LOG(INFO) << "After init: [shm_malloc] " << shm_malloc_.GetInfo();
  }

  void Get(const uint64_t key, std::string &value, unsigned tid) override {
    // TODO:
    // 1. get <key, pointer> from index
    // 2. get value length from value header
    // 2. get value from pointer and length

    base::PetKVData shmkv_data;
    // std::shared_lock<std::shared_mutex> _(lock_);
    Key_t hash_key = key;
    Value_t read_value = index->Get(hash_key);

    if (read_value == NONE) {
      value = std::string();
    } else {
      shmkv_data.data_value = read_value;
      char *data = shm_malloc_.GetMallocData(shmkv_data.shm_malloc_offset());
      if (data == nullptr) {
        value = std::string();
        return;
      }
#ifdef XMH_VARIABLE_SIZE_KV
      int size = shm_malloc_.GetMallocSize(shmkv_data.shm_malloc_offset());
#else
      int size = value_size_;
#endif
      value = std::string(data, size);
    }
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned tid) override {
    // TODO: 
    // 1. malloc space for value
    //     -> value length in value header
    //     -> return the unified pointer to value
    // 2. insert <key, pointer> into index

    base::PetKVData shmkv_data;
    char *sync_data = shm_malloc_.New(value.size());
    shmkv_data.SetShmMallocOffset(shm_malloc_.GetMallocOffset(sync_data));
    memcpy(sync_data, value.data(), value.size());

    Key_t hash_key = key;
    index->Insert(hash_key, shmkv_data.data_value);
  }

  void BatchGet(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>> *values,
                unsigned tid) override {
    values->clear();
    // std::shared_lock<std::shared_mutex> _(lock_);

    for (auto k : keys) {
      base::PetKVData shmkv_data;
      Key_t hash_key = k;
      Value_t read_value = index->Get(hash_key);

      if (read_value == NONE) {
        values->emplace_back();
      } else {
        shmkv_data.data_value = read_value;
        char *data = shm_malloc_.GetMallocData(shmkv_data.shm_malloc_offset());
        if (data == nullptr) {
          values->emplace_back();
          continue;
        }
#ifdef XMH_VARIABLE_SIZE_KV
        int size = shm_malloc_.GetMallocSize(shmkv_data.shm_malloc_offset());
#else
        int size = value_size_;
#endif
        values->emplace_back((float *)data, size / sizeof(float));
      }
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
  Index *index;
  // std::shared_mutex lock_;

  uint64_t counter = 0;
  std::string dict_pool_name_;
  size_t dict_pool_size_;
  int value_size_;
#ifdef XMH_SIMPLE_MALLOC
  base::PersistSimpleMalloc shm_malloc_;
#else
  base::PersistLoopShmMalloc shm_malloc_;
#endif
  base::ShmFile valid_shm_file_;
};

FACTORY_REGISTER(BaseKV, KVEngineHybrid, KVEngineHybrid,
                 const BaseKVConfig &);