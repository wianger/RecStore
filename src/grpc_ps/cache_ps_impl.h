#pragma once
#include <algorithm>
#include <atomic>
#include <boost/coroutine2/all.hpp>
#include <cstdint>
#include <experimental/filesystem>

#include "base/array.h"
#include "base/factory.h"
#include "base/log.h"  // NOLINT
#include "base/timer.h"
#include "folly/ProducerConsumerQueue.h"
#include "parameters.h"
#include "storage/kv_engine/base_kv.h"

using boost::coroutines2::coroutine;

static const int KEY_CNT = 12543670;

template <typename key_t>
struct TaskElement {
  TaskElement(const base::ConstArray<key_t> &keys,
              const base::MutableArray<ParameterPack> &packs,
              std::atomic_bool *promise)
      : keys(keys), packs(packs), promise(promise) {}

  TaskElement() {}

  base::ConstArray<key_t> keys;
  base::MutableArray<ParameterPack> packs;
  std::atomic_bool *promise;
};

class CachePS {
 public:
  using key_t = uint64_t;

  CachePS(int64_t dict_capability, int value_size, int64_t memory_pool_size,
          int num_threads, int corotine_per_thread, int max_batch_keys_size)
      : value_size(value_size) {
    BaseKVConfig config;
    config.num_threads_ = num_threads;
    config.json_config_["capacity"] = dict_capability;
    config.json_config_["value_size"] = value_size;
    config.json_config_["memory_pool_size"] = memory_pool_size;
    config.json_config_["corotine_per_thread"] = corotine_per_thread;
    config.json_config_["max_batch_keys_size"] = max_batch_keys_size;
    auto p = base::Factory<BaseKV, const BaseKVConfig &>::NewInstance(
        "KVEngineMap", config);
    base_kv_.reset(p);
  }

  ~CachePS() {}

  bool Initialize(const std::vector<std::string> &model_config_path,
                  const std::vector<std::string> &emb_file_path) {
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() { base_kv_->clear(); }

  void LoadFakeData(int64_t key_size) {
    std::vector<uint64_t> keys;
    float *values = new float[value_size / sizeof(float) * key_size];
    for (int64_t i = 0; i < key_size; i++) {
      keys.push_back(i);
    }
    base_kv_->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  }

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path) {
    // base_kv_->loadCkpt();
    // LoadFakeData(KEY_CNT);
    return true;
  }

  void PutSingleParameter(const ParameterCompressItem *item, int tid) {
    bool success;
    auto key = item->key;
    auto dim = item->dim;
    base_kv_->Put(
        key, std::string_view((char *)item->data(), dim * sizeof(float)), tid);
  }

  void PutParameter(coroutine<void>::push_type &sink,
                    const ParameterCompressReader *reader, int tid) {
    std::vector<uint64_t> keys_vec;
    std::vector<base::ConstArray<float>> values;
    for (int i = 0; i < reader->item_size(); i++) {
      keys_vec.emplace_back(reader->item(i)->key);
      values.emplace_back((float *)reader->item(i)->data(),
                          reader->item(i)->dim);
    }
    base::ConstArray<uint64_t> keys(keys_vec);

    base_kv_->BatchPut(sink, keys, &values, tid);
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack &pack, int tid) {
    std::vector<uint64_t> keys = {key};
    base::ConstArray<uint64_t> keys_array(keys);
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(keys_array, &values, tid);
    base::ConstArray<float> value = values[0];

    if (value.Size() == 0) {
      pack.key = key;
      pack.dim = 0;
      pack.emb_data = nullptr;
      FB_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key = key;
    pack.dim = value.Size();
    pack.emb_data = value.Data();
    // LOG(ERROR) << "Get key " << key << " dim " << pack.dim;
    return true;
  }

  bool GetParameterRun2Completion(coroutine<void>::push_type &sink,
                                  base::ConstArray<uint64_t> keys,
                                  std::vector<ParameterPack> &pack, int tid) {
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(sink, keys, &values, tid);

    for (int i = 0; i < keys.Size(); i++) {
      pack.emplace_back(keys[i], values[i].Size(), values[i].Data());
    }

    return true;
  }

 private:
  int value_size;
  std::unique_ptr<BaseKV> base_kv_;
  std::atomic<bool> stopFlag_{false};
};