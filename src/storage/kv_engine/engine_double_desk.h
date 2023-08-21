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
  char *cache_;
  uint64_t cache_size;
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
  static const int MAX_QUEUE_NUM = 32;
  static constexpr int kBouncedBuffer_ = 20000;
  uint64_t vector_capability;
  char *rawbouncedBuffer_;
  char *rawwrite_buffer_;
  char *bouncedBuffer_[MAX_QUEUE_NUM];
  char *write_buffer_[MAX_QUEUE_NUM];
  std::unique_ptr<ssdps::SpdkWrapper> ssd_;

  std::pair<int64_t, int> Mapping(int64_t index) const {
#if 0
    int64_t lba_no = index * VALUE_SIZE / ssd_->GetLBASize();
    int in_lba_offset = (index * VALUE_SIZE) % ssd_->GetLBASize();
    return std::make_pair(lba_no, in_lba_offset);
#else 
#if 0
    int64_t lba_no = index * 1;
    int in_lba_offset = 0;
    return std::make_pair(lba_no, in_lba_offset);
#else
    uint64_t lba_no = ssd_->GetLBANumber() * index / vector_capability;
    int in_lba_offset = 0;
    return std::make_pair(lba_no, in_lba_offset);
#endif
#endif
  }

  static void ReadCompleteCB(void *ctx, const struct spdk_nvme_cpl *cpl) {
    std::atomic<int> *read_complete = (std::atomic<int> *)ctx;
    if (FOLLY_UNLIKELY(spdk_nvme_cpl_is_error(cpl))) {
      LOG(FATAL) << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }
    read_complete->fetch_add(1);
  }

  static void BulkLoadCB(void *ctx, const struct spdk_nvme_cpl *cpl) {
    if (FOLLY_UNLIKELY(spdk_nvme_cpl_is_error(cpl))) {
      LOG(FATAL) << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }
    std::atomic_int *counter = (std::atomic_int *)ctx;
    counter->fetch_add(1);
  }

  void SubBulkLoad(int keys_size,
                   base::ConstArray<uint64_t> indexs_array, const void *value,
                   char *pinned_value, int tid) {
    CHECK(keys_size == indexs_array.Size());

    int64_t subarray_size = keys_size;

    std::atomic_int finished_counter{0};  // # of finished write page
    int submit_counter = 0;               // # of all writed pages
    int64_t old_page_id = -1;
    for (int64_t i = 0; i < subarray_size; i++) {
      uint64_t index = indexs_array[i];
      CHECK_LT(Mapping(index).second, ssd_->GetLBASize());
      CHECK_GE(Mapping(index).second, 0);
      if (old_page_id != -1 && old_page_id != Mapping(index).first) {
        // write page
        int ret;
        do {
          ret = ssd_->SubmitWriteCommand(
              pinned_value + submit_counter * ssd_->GetLBASize(),
              ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter, tid);
          ssd_->PollCompleteQueue(tid);
        } while (ret != 0);
        submit_counter++;
      }
      memcpy(pinned_value + submit_counter * ssd_->GetLBASize() +
                 Mapping(index).second,
             (char *)value + i * value_size, value_size);
      old_page_id = Mapping(index).first;
    }
    // write the last page
    int ret;
    do {
      ret = ssd_->SubmitWriteCommand(
          pinned_value + submit_counter * ssd_->GetLBASize(),
          ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter, tid);
      ssd_->PollCompleteQueue(tid);
    } while (ret != 0);
    submit_counter++;
    while (submit_counter != finished_counter) ssd_->PollCompleteQueue(tid);
  }

  void SubBulkLoad(int keys_size, base::ConstArray<uint64_t> indexs_array, std::vector<base::ConstArray<float>> &value, int start,
                   char *pinned_value, int tid) {
    CHECK(keys_size == indexs_array.Size());

    int64_t subarray_size = keys_size;

    std::atomic_int finished_counter{0};  // # of finished write page
    int submit_counter = 0;               // # of all writed pages
    int64_t old_page_id = -1;
    for (int64_t i = 0; i < subarray_size; i++) {
      uint64_t index = indexs_array[i];
      CHECK_LT(Mapping(index).second, ssd_->GetLBASize());
      CHECK_GE(Mapping(index).second, 0);
      if (old_page_id != -1 && old_page_id != Mapping(index).first) {
        // write page
        int ret;
        do {
          ret = ssd_->SubmitWriteCommand(
              pinned_value + submit_counter * ssd_->GetLBASize(),
              ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter, tid);
          ssd_->PollCompleteQueue(tid);
        } while (ret != 0);
        submit_counter++;
      }
      memcpy(pinned_value + submit_counter * ssd_->GetLBASize() +
                 Mapping(index).second,
             value[i + start].Data(), value_size);
      old_page_id = Mapping(index).first;
    }
    // write the last page
    int ret;
    do {
      ret = ssd_->SubmitWriteCommand(
          pinned_value + submit_counter * ssd_->GetLBASize(),
          ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter, tid);
      ssd_->PollCompleteQueue(tid);
    } while (ret != 0);
    submit_counter++;
    while (submit_counter != finished_counter) ssd_->PollCompleteQueue(tid);
  }

public:
  explicit KVEngineDoubleDesk(const BaseKVConfig &config)
    : BaseKV(config), 
    cache_size(config.capacity * config.value_size * 0.05),
    value_size(config.value_size),
    max_batch_keys_size(config.max_batch_keys_size),
    per_thread_buffer_size(value_size * max_batch_keys_size),
    thread_num(config.num_threads),
    cache_entry_size(cache_size / value_size),
    lf_list(cache_entry_size),
    vector_capability(config.capacity)
    {

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
    
    ssd_ = ssdps::SpdkWrapper::create(thread_num);
    CHECK(ssd_) << "failed to allocate ssd";

    cache_ = new char[cache_size];
    CHECK(cache_) << "failed to allocate cache";
    
    for (int i = 0; i < thread_num; i++) {
      per_thread_buffer[i] = new char[per_thread_buffer_size];
      unhit_array[i] = new uint64_t[max_batch_keys_size];
    }
    Init();
    ssd_->Init();
    rawbouncedBuffer_ = (char *)spdk_malloc(kBouncedBuffer_ * ssd_->GetLBASize() * thread_num, 0, NULL,
                                 SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
                            
    // cudaMallocHost(&bouncedBuffer_, kBouncedBuffer_ * ssd_->GetLBASize(),
    //                cudaHostAllocDefault);
    CHECK(rawbouncedBuffer_);
    const int nr_batch_pages = 32;
    int64_t pinned_bytes = ssd_->GetLBASize() * nr_batch_pages;
    rawwrite_buffer_ = (char *)spdk_malloc(
        pinned_bytes * thread_num, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK(rawwrite_buffer_) << "spdk_malloc";

    for (int i = 0; i < thread_num; i++) {
      bouncedBuffer_[i] = (char *)rawbouncedBuffer_ + i * kBouncedBuffer_ * ssd_->GetLBASize();
      write_buffer_[i] = (char *)rawwrite_buffer_ + i * pinned_bytes;
    }
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
    spdk_free(rawwrite_buffer_);
    spdk_free(rawbouncedBuffer_);
    delete cache_;
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

    std::atomic<int> readCompleteCount{0};
    xmh::Timer timer_kvell_submitCommand("Hier-SSD command");
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
      } else {
        int64_t count_offset = -1;
        count_offset = key_iter->second;
        timer_kvell_submitCommand.CumStart();
        CHECK_LE(value_size, ssd_->GetLBASize()) << "KISS";
        int64_t lba_no;
        int in_lba_offset;
        std::tie(lba_no, in_lba_offset) = Mapping(count_offset);
        ssd_->SubmitReadCommand(bouncedBuffer_[t] + i * ssd_->GetLBASize(),
                                value_size, lba_no, ReadCompleteCB, &readCompleteCount, t);
        timer_kvell_submitCommand.CumEnd();

        values->emplace_back(
            (float *)(bouncedBuffer_[t] + i * ssd_->GetLBASize() + in_lba_offset),
            value_size / sizeof(float));
        unhit_array[t][unhit_size] = key_iter->second;
        unhit_size++;
      }
    }
    timer_kvell_submitCommand.CumReport();
    index_timer.CumEnd();
    ssd_timer.CumStart();
    xmh::PerfCounter::Record("unhit_size Keys", unhit_size);
    while(unhit_size != readCompleteCount.load()){
      ssd_->PollCompleteQueue(t);
    }
    ssd_timer.CumEnd();
    cache_timer.CumStart();
    auto free_pos = lf_list.TryPop(unhit_size);
    int j = 0;
    for(int i = free_pos.first; i != free_pos.second; i = (i + 1) % cache_entry_size){
      index_info[unhit_array[t][j]].in_cache = true;
      int pos = lf_list[i];
      index_info[unhit_array[t][j]].cache_offset = pos;
      int64_t lba_no;
      int in_lba_offset;
      std::tie(lba_no, in_lba_offset) = Mapping(unhit_array[t][j]);
      memcpy(cache_ + pos * value_size, bouncedBuffer_[t] + i * ssd_->GetLBASize() + in_lba_offset, value_size);
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
    BatchPutSSD(base::ConstArray<uint64_t>(keys_arr), values, t);
  }

  void BatchPutSSD(base::ConstArray<uint64_t> keys_array, std::vector<base::ConstArray<float>> &value, int tid) {
    const int nr_batch_pages = 32;
    int i = 0;
    while(i < keys_array.Size()) {
      int batched_size = std::min(nr_batch_pages, keys_array.Size() - i);
      SubBulkLoad(
          batched_size,
          keys_array.SubArray(i, i + batched_size), value, i,
          write_buffer_[tid], tid);
      i += batched_size;
    }
  }

  void Put(const uint64_t key, const std::string_view &value,
           unsigned t) override {
    CHECK(0) << "not implemented";
  }

  void clear() override {
    Init();
    BulkLoad({}, nullptr);
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