#pragma once
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <experimental/filesystem>

#include "folly/GLog.h"  // NOLINT
#include "folly/ProducerConsumerQueue.h"
// #include "inference/inference_utils.hpp"
#include "base/timer.h"
#include "base/array.h"
// #include "nlohmann/json.hpp"
#include "parameters.h"
// #include "parser.hpp"

#define XMH_ROBIN_HOOD_HASH

#ifdef XMH_FOLLY_HASH
#include "folly/AtomicHashArray.h"  // NOLINT
#endif

#ifdef XMH_STD_HASH
#include <unordered_map>
#endif

#ifdef XMH_ROBIN_HOOD_HASH
#include "robin_hood.h"
#endif

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
#ifdef XMH_FOLLY_HASH
  using dict_t = folly::AtomicHashArray<key_t, ParameterCompressItem *>;
#endif
#ifdef XMH_STD_HASH
  using dict_t = std::unordered_map<key_t, ParameterCompressItem *>;
#endif
#ifdef XMH_ROBIN_HOOD_HASH
  using dict_t = robin_hood::unordered_map<key_t, ParameterCompressItem *>;
#endif

  using CPUCacheGetTaskQ = TaskElement<key_t>;

  CachePS(int64_t dict_capability, int64_t value_size) {
    // dict_.reset(dict_t::create(dict_capability).release());
    dict_ = std::make_unique<dict_t>();
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
    // Clear();
    LOG(INFO) << "Before Load CKPT";
    LoadCkpt(model_config_path, emb_file_path);
    LOG(INFO) << "After Load CKPT";
    return true;
  }

  void Clear() {
    LOG(WARNING) << "Clean ps";
    for (auto &pair : *dict_) {
      free(pair.second);
    }
    dict_->clear();
  }

  static bool strVectorEq(const std::vector<std::string> &a,
                          const std::vector<std::string> &b) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }

  bool LoadCkpt(const std::vector<std::string> &model_config_path,
                const std::vector<std::string> &emb_file_path) {
    static std::vector<std::string> previous_model_config_path;
    static std::vector<std::string> previous_emb_file_path;
    if (strVectorEq(previous_model_config_path, model_config_path) &&
        strVectorEq(previous_emb_file_path, emb_file_path)) {
      return true;
    }
    Clear();
    previous_model_config_path = model_config_path;
    previous_emb_file_path = emb_file_path;

    CHECK_EQ(model_config_path.size(), 1);
    // Initialize for each model
    ps_config_ = HugeCTR::parameter_server_config();
    for (unsigned int i = 0; i < model_config_path.size(); i++) {
      // Open model config file and input model json config
      nlohmann::json model_config(
          HugeCTR::read_json_file(model_config_path[i]));

      ps_config_.emb_file_name_.emplace_back(emb_file_path);

      // Read embedding layer config
      const nlohmann::json &j_layers =
          HugeCTR::get_json(model_config, "layers");
      std::vector<bool> distributed_emb;
      std::vector<size_t> embedding_vec_size;
      std::vector<float> default_emb_vec_value;
      // Search for all embedding layers
      for (unsigned int j = 1; j < j_layers.size(); j++) {
        const nlohmann::json &j_single_layer = j_layers[j];
        std::string embedding_type =
            HugeCTR::get_value_from_json<std::string>(j_single_layer, "type");
        if (embedding_type.compare("DistributedSlotSparseEmbeddingHash") == 0) {
          distributed_emb.emplace_back(true);
          const nlohmann::json &embedding_hparam =
              HugeCTR::get_json(j_single_layer, "sparse_embedding_hparam");
          embedding_vec_size.emplace_back(HugeCTR::get_value_from_json<size_t>(
              embedding_hparam, "embedding_vec_size"));
          default_emb_vec_value.emplace_back(
              HugeCTR::get_value_from_json_soft<float>(
                  embedding_hparam, "default_emb_vec_value", 0.0f));
        } else if (embedding_type.compare("LocalizedSlotSparseEmbeddingHash") ==
                       0 ||
                   embedding_type.compare(
                       "LocalizedSlotSparseEmbeddingOneHot") == 0) {
          distributed_emb.emplace_back(false);
          const nlohmann::json &embedding_hparam =
              HugeCTR::get_json(j_single_layer, "sparse_embedding_hparam");
          embedding_vec_size.emplace_back(HugeCTR::get_value_from_json<size_t>(
              embedding_hparam, "embedding_vec_size"));
          default_emb_vec_value.emplace_back(
              HugeCTR::get_value_from_json_soft<float>(
                  embedding_hparam, "default_emb_vec_value", 0.0f));
        } else {
          break;
        }
      }
      ps_config_.distributed_emb_.emplace_back(distributed_emb);
      ps_config_.embedding_vec_size_.emplace_back(embedding_vec_size);
      ps_config_.default_emb_vec_value_.emplace_back(default_emb_vec_value);
    }

    if (ps_config_.distributed_emb_.size() != model_config_path.size() ||
        ps_config_.embedding_vec_size_.size() != model_config_path.size() ||
        ps_config_.default_emb_vec_value_.size() != model_config_path.size()) {
      LOG(FATAL)
          << "Wrong input: The size of parameter server parameters are not "
             "correct.";
    }
    // Load embeddings for each embedding table from each model
    for (unsigned int i = 0; i < model_config_path.size(); i++) {
      size_t num_emb_table = (ps_config_.emb_file_name_[i]).size();
      // Temp vector of embedding table for this model
      for (unsigned int j = 0; j < num_emb_table; j++) {
        // Create input file stream to read the embedding file
        const std::string emb_file_prefix = ps_config_.emb_file_name_[i][j];
        const std::string key_file = emb_file_prefix + "/key";
        const std::string vec_file = emb_file_prefix + "/emb_vector";
        std::ifstream key_stream(key_file);
        std::ifstream vec_stream(vec_file);
        // Check if file is opened successfully
        if (!key_stream.is_open() || !vec_stream.is_open()) {
          LOG(FATAL) << "Error: embeddings file not open for reading"
                     << key_file << "; " << vec_file;
        }
        size_t key_file_size_in_byte = fs::file_size(key_file);
        size_t vec_file_size_in_byte = fs::file_size(vec_file);

        size_t key_size_in_byte = sizeof(key_t);
        size_t vec_size_in_byte =
            sizeof(float) * ps_config_.embedding_vec_size_[i][j];

        size_t num_key = key_file_size_in_byte / key_size_in_byte;
        size_t num_vec = vec_file_size_in_byte / vec_size_in_byte;
        if (num_key != num_vec) {
          LOG(FATAL) << "Error: num_key != num_vec in embedding file";
        }
        size_t num_float_val_in_vec_file =
            vec_file_size_in_byte / sizeof(float);

        // The temp embedding table
        std::vector<key_t> key_vec(num_key, 0);
        std::vector<float> vec_vec(num_float_val_in_vec_file, 0.0f);
        key_stream.read(reinterpret_cast<char *>(key_vec.data()),
                        key_file_size_in_byte);
        vec_stream.read(reinterpret_cast<char *>(vec_vec.data()),
                        vec_file_size_in_byte);

        const size_t emb_vec_size = ps_config_.embedding_vec_size_[i][j];
        LOG(INFO) << "load from " << emb_file_prefix << " ; # of keys is "
                  << num_key;
        for (size_t i = 0; i < num_key; i++) {
          FB_LOG_EVERY_MS(INFO, 1000)
              << "Loading Ckpt " << (float)i / num_key * 100 << "%";

          ParameterCompressItem *value = (ParameterCompressItem *)malloc(
              ParameterCompressItem::GetSize(emb_vec_size));
          value->key = key_vec[i];
          value->dim = emb_vec_size;
          std::copy_n(vec_vec.begin() + i * emb_vec_size, emb_vec_size,
                      value->embedding);
          // TODO: remove me
          // std::fill_n(value->embedding, emb_vec_size, value->key);

          bool success;
          std::tie(std::ignore, success) =
              dict_->insert(std::make_pair(key_vec[i], value));
          CHECK(success);
        }
      }
    }
    return true;
  }

  void PutSingleParameter(const ParameterCompressItem *item) {
    bool success;
    auto key = item->key;
    auto dim = item->dim;

    ParameterCompressItem *value =
        (ParameterCompressItem *)malloc(ParameterCompressItem::GetSize(dim));
    value->key = key;
    value->dim = dim;
    std::copy_n(item->data(), dim, value->embedding);
    std::tie(std::ignore, success) = dict_->insert(std::make_pair(key, value));
    CHECK(success);
  }

  bool GetParameterRun2Completion(key_t key, ParameterPack &pack) {
#ifdef XMH_FOLLY_HASH
    FB_LOG_EVERY_MS(INFO, 1000)
        << "load factor = " << (double)dict_->size() / dict_->capacity_;
#endif

#if defined(XMH_ROBIN_HOOD_HASH) || defined(XMH_STD_HASH)
    FB_LOG_EVERY_MS(INFO, 1000) << "load factor = " << dict_->load_factor();
#endif
    auto ret = dict_->find(key);
    if (ret == dict_->end()) {
      pack.key = key;
      pack.dim = 0;
      pack.emb_data = nullptr;
      FB_LOG_EVERY_MS(ERROR, 1000) << "key " << key << " not existing";
      return false;
    }
    pack.key = key;
    pack.dim = ret->second->dim;
    pack.emb_data = ret->second->data();
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

#ifdef XMH_FOLLY_HASH
  dict_t::SmartPtr dict_
#endif
#if defined(XMH_ROBIN_HOOD_HASH) || defined(XMH_STD_HASH)
      std::unique_ptr<dict_t>
          dict_;
#endif

  static constexpr int get_thread_num_ = 2;
  static_assert(get_thread_num_ < 10);
  HugeCTR::parameter_server_config ps_config_;
  std::atomic<bool> stopFlag_{false};
  std::vector<std::thread> get_threads_;
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<CPUCacheGetTaskQ>>>
      getTaskQs_;
};