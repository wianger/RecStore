#pragma once
#include <folly/container/F14Map.h>
#include <memory>

#include "base/factory.h"
#include "base_kv.h"
#include "base/lf_list.h"
#include "storage/ssd/naiveKVell.h"

DECLARE_int32(prefetch_method);

class KVEngineDoubleDesk : public BaseKV {
  struct IndexInfo{
    bool in_cache = false;
    int cache_offset = -1;
    uint64_t hit_cnt;
  };

  using dict_type = folly::F14FastMap<
      uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
      folly::f14::DefaultAlloc<std::pair<uint64_t const, uint64_t>>>;

  constexpr static uint64_t MAX_THREAD_CNT = 32;
  ssdps::NaiveArraySSD<uint64_t> *ssd_;
  char *cache_;
  uint64_t CACHE_SIZE = 1l << 30;

  uint64_t value_size;
  uint64_t max_batch_keys_size;
  uint64_t per_thread_buffer_size;
  dict_type hash_table_;
  uint64_t *unhit_array[MAX_THREAD_CNT];
  char *per_thread_buffer[MAX_THREAD_CNT];
  IndexInfo *index_info;
  int thread_num;
  uint64_t key_cnt;
  uint64_t cache_entry_size;
  base::LFList lf_list;

public:
  explicit KVEngineDoubleDesk(const BaseKVConfig &config)
    : BaseKV(config), 
    value_size(config.value_size),
    max_batch_keys_size(config.max_batch_keys_size),
    per_thread_buffer_size(value_size * max_batch_keys_size),
    thread_num(config.num_threads),
    cache_entry_size(CACHE_SIZE / value_size),
    lf_list(cache_entry_size) {

    CHECK(value_size % sizeof(float) == 0) << "value_size must be multiple of 4";
    CHECK_GT(value_size, 0) << "value_size must be positive";
    CHECK_GT(max_batch_keys_size, 0) << "max_batch_keys_size must be positive";
    CHECK_GT(per_thread_buffer_size, 0) << "per_thread_buffer_size must be positive";
    CHECK_GT(thread_num, 0) << "thread_num must be positive";
    CHECK_LE(thread_num, MAX_THREAD_CNT) << "thread_num must be less than " << MAX_THREAD_CNT;
    LOG(INFO) << "value_size: " << value_size;
    LOG(INFO) << "max_batch_keys_size: " << max_batch_keys_size;
    LOG(INFO) << "per_thread_buffer_size: " << per_thread_buffer_size;
    LOG(INFO) << "thread_num: " << thread_num;

    index_info = new IndexInfo[config.capacity];
    CHECK(index_info) << "failed to allocate index_info";
    
    ssd_ = new ssdps::NaiveArraySSD<uint64_t>(config.value_size, config.capacity, thread_num);
    CHECK(ssd_) << "failed to allocate ssd";

    cache_ = new char[CACHE_SIZE];
    CHECK(cache_) << "failed to allocate cache";
    
    for (int i = 0; i < thread_num; i++) {
      per_thread_buffer[i] = new char[per_thread_buffer_size];
      unhit_array[i] = new uint64_t[max_batch_keys_size];
    }
    Init();
  }

  void Init(){
    hash_table_.clear();
    lf_list.clear();
    std::vector<int> free_list;
    for(int i = 0; i < cache_entry_size; i++){
      free_list.push_back(i);
    }
    lf_list.InsertFreeList(free_list);
    key_cnt = 0;
  }

  ~KVEngineDoubleDesk() override {
    for (int i = 0; i < thread_num; i++) {
      delete per_thread_buffer[i];
      delete unhit_array[i];
    }
    delete cache_;
    delete ssd_;
    delete index_info;
  }

  void Get(const uint64_t key, std::string &value, unsigned t) override {
    LOG(FATAL) << "not implemented";
  }

  void BatchGet(base::ConstArray<uint64> keys,
                std::vector<base::ConstArray<float>> *values,
                unsigned t) override {
    xmh::Timer index_timer("BatchGet index");
    xmh::Timer ssd_timer("BatchGet ssd");
    xmh::Timer cache_timer("BatchGet cache");
    int unhit_size = 0;
    index_timer.CumStart();
    for (int i = 0; i < keys.Size(); i++) {
      const auto key_iter = hash_table_.find(keys[i]);
      if(key_iter == hash_table_.end()){
        values->emplace_back(nullptr, 0);
        continue;
      }
      IndexInfo *info = &index_info[key_iter->second];
      if(info->in_cache){
        info->hit_cnt++;
        values->emplace_back((float *)(cache_ + info->cache_offset * value_size), value_size / sizeof(float));
        continue;
      }
      values->emplace_back(
          (float *)(per_thread_buffer[t] + unhit_size * value_size),
          value_size / sizeof(float));
      unhit_array[t][unhit_size] = key_iter->second;
      unhit_size++;
    }
    index_timer.CumEnd();
    ssd_timer.CumStart();
    xmh::PerfCounter::Record("unhit_size Keys", unhit_size);
    if (unhit_size != 0) {
      base::ConstArray<uint64> unhit_keys(unhit_array[t], unhit_size);
      ssd_->BatchGet(unhit_keys, base::ConstArray<uint64_t>(),
                     per_thread_buffer[t], t);
    }
    ssd_timer.CumEnd();
    cache_timer.CumStart();
    auto free_pos = lf_list.TryPop(unhit_size);
    int j = 0;
    for(int i = free_pos.first; i != free_pos.second; i = (i + 1) % cache_entry_size){
      index_info[unhit_array[t][j]].in_cache = true;
      int pos = lf_list[i];
      index_info[unhit_array[t][j]].cache_offset = pos;
      memcpy(cache_ + pos * value_size, per_thread_buffer[t] + j * value_size, value_size);
      j++;
    }
    cache_timer.CumEnd();
    index_timer.CumReport();
    ssd_timer.CumReport();
    cache_timer.CumReport();
  }

  void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) override {
    Init();
    key_cnt = keys.Size();
    for (int i = 0; i < keys.Size(); i++) {
      hash_table_[keys[i]] = i;
      index_info[i].in_cache = false;
    }
    // ssd_->BulkLoad(keys.Size(), value);
  }

  std::pair<uint64_t, uint64_t> RegisterPMAddr() const override {
    return std::make_pair(0, 0);
  }

  void BatchPut(base::ConstArray<uint64_t> keys,
                std::vector<base::ConstArray<float>> &values,
                unsigned t) override {
    std::vector<uint64_t> keys_arr;
    for(int i = 0; i < keys.Size(); i++){
      auto &key = keys[i];
      auto iter = hash_table_.find(key);
      uint64_t index_pos = -1;
      if(iter == hash_table_.end()){
        index_pos = key_cnt++;
        hash_table_[key] = index_pos;
        index_info[index_pos].in_cache = false;
        index_info[index_pos].hit_cnt = 0;
      } else {
        index_pos = iter->second;
      }
      IndexInfo *info = &index_info[index_pos];
      keys_arr.emplace_back(index_pos);
      if(info->in_cache){
        memcpy(cache_ + info->cache_offset * value_size, values[i].Data(), value_size);
      }
    }
    ssd_->BatchPut(base::ConstArray<uint64_t>(keys_arr), values, t);
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned t) override {
    auto iter = hash_table_.find(key);
    uint64_t index_pos = -1;
    if(iter == hash_table_.end()){
      index_pos = key_cnt++;
      hash_table_[key] = index_pos;
      index_info[index_pos].in_cache = false;
      index_info[index_pos].hit_cnt = 0;
    } else {
      index_pos = iter->second;
    }
    IndexInfo *info = &index_info[index_pos];
    std::vector<uint64_t> keys_arr{index_pos};
    ssd_->BatchPut(base::ConstArray<uint64_t>(keys_arr), value.data(), t);
    if(info->in_cache){
      memcpy(cache_ + info->cache_offset * value_size, value.data(), value_size);
    }
  }

  void clear() override {
    Init();
    ssd_->BulkLoad(0, nullptr);
  }

  void Cleaner() {
    std::vector<int> indexes;
    std::vector<int> free_list;
    uint64_t hit_cnt_sum;
    while(1){
      indexes.clear();
      free_list.clear();
      hit_cnt_sum = 0;
      for (int i = 0; i < key_cnt; i++) {
        if (index_info[i].in_cache) {
          indexes.push_back(i);
          hit_cnt_sum += index_info[i].hit_cnt;
        }
      }
      uint64_t half_hit_cnt_avg = hit_cnt_sum / indexes.size() / 2;
      for(int i = 0; i < indexes.size(); i++){
        int index = indexes[i];
        if(index_info[index].hit_cnt < half_hit_cnt_avg){
          free_list.push_back(index_info[index].cache_offset);
          index_info[index].in_cache = false;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
      lf_list.InsertFreeList(free_list);
    }
  }
};

FACTORY_REGISTER(BaseKV, KVEngineDoubleDesk, KVEngineDoubleDesk,
                 const BaseKVConfig &);