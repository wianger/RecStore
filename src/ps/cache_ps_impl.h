#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <experimental/filesystem>

#include "base/array.h"
#include "base/timer.h"
#include "base/log.h"  // NOLINT
#include "base/factory.h"

#include "storage/kv_engine/base_kv.h"
#include "folly/ProducerConsumerQueue.h"
#include "parameters.h"

DECLARE_bool(use_master_worker);

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

  using CPUCacheGetTaskQ = TaskElement<key_t>;

  CachePS(int64_t dict_capability, int64_t value_size, int num_threads) {
    BaseKVConfig config;
    config.path = "/dev/shm/double-placeholder";
    config.capacity = dict_capability;
    config.value_size = value_size;
    config.num_threads = num_threads;

    auto p = base::Factory<BaseKV, const BaseKVConfig &>::NewInstance("KVEngineDash",
                                                                      config);

    base_kv_.reset(p);

    if (FLAGS_use_master_worker) {
      for (int worker_id = 0; worker_id < get_thread_num_; worker_id++) {
        getTaskQs_.push_back(nullptr);
        getTaskQs_[worker_id].reset(
            new folly::ProducerConsumerQueue<CPUCacheGetTaskQ>(64));
        get_threads_.emplace_back(&CachePS::GetPollingThread, this, worker_id);
      }
      LOG(INFO) << "use multi thread get, thread num is " << get_thread_num_;
    } else {
      LOG(INFO) << "use single thread get";
    }
  }

  ~CachePS() {
    if (FLAGS_use_master_worker) {
      stopFlag_ = true;
      for (auto &each : get_threads_) {
        each.join();
      }
    }
  }

  bool Initialize(const std::vector<std::string> &model_config_path,
                  const std::vector<std::string> &emb_file_path) {
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() { base_kv_->clear(); }

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path) {
    return true;
  }

  void PutSingleParameter(const ParameterCompressItem *item) {
    bool success;
    auto key = item->key;
    auto dim = item->dim;

    base_kv_->Put(
        key, std::string_view((char *)item->data(), dim * sizeof(float)), 0);
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack &pack) {
    std::vector<uint64_t> keys = {key};
    base::ConstArray<uint64_t> keys_array(keys);
    std::vector<base::ConstArray<float>> values;

    base_kv_->BatchGet(keys_array, &values, 0);
    base::ConstArray<float> value = values[0];


    if (value.Size() == 0) {
      pack.key = key;
      pack.dim = 0;
      pack.emb_data = nullptr;
      FB_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key = key;
    pack.dim = value.Size() / sizeof(float);
    pack.emb_data = value.Data();
    return true;
  }

  void GetPollingThread(int worker_id) {
    CPUCacheGetTaskQ getTaskElement;
    auto q = getTaskQs_[worker_id].get();
    while (!stopFlag_.load()) {
      if (!q->read(getTaskElement)) continue;
      auto &keys = getTaskElement.keys;
      for (int i = 0; i < keys.Size(); i++) {
        GetParameterRun2Completion(keys[i], getTaskElement.packs[i]);
      }
      getTaskElement.promise->store(true);
    }
  }

  bool GetParameterMasterWorker(
      const base::ConstArray<key_t> &keys,
      std::vector<std::vector<ParameterPack>> *packs) {
    CHECK(FLAGS_use_master_worker);
    packs->resize(get_thread_num_);

    int size = keys.Size();
    int worksPerThread = size / get_thread_num_ + 1;

    for (int i = 0; i < get_thread_num_; i++) {
      int start = i * worksPerThread;
      int end = std::min(size, (i + 1) * worksPerThread);

      get_thread_promises_[i].promise = false;
      base::ConstArray<key_t> sub_keys(&keys[start], end - start);
      packs->at(i).resize(end - start);
      base::MutableArray<ParameterPack> sub_packs(packs->at(i));
      while (!getTaskQs_[i]->write(sub_keys, sub_packs,
                                   &get_thread_promises_[i].promise))
        ;
    }
    for (int i = 0; i < get_thread_num_; i++) {
      while (!get_thread_promises_[i].promise.load())
        ;
    }
    return true;
  }

 private:
  union {
    std::atomic_bool promise;
    char _[64];
  } get_thread_promises_[10];

  std::unique_ptr<BaseKV> base_kv_;

  static constexpr int get_thread_num_ = 2;
  static_assert(get_thread_num_ < 10);
  std::atomic<bool> stopFlag_{false};
  std::vector<std::thread> get_threads_;
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<CPUCacheGetTaskQ>>>
      getTaskQs_;
};