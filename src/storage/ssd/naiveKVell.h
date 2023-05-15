#pragma once
#include <folly/GLog.h>
#include <folly/Portability.h>
#include <folly/system/MemoryMapping.h>

#include <boost/algorithm/string/join.hpp>
#include <experimental/filesystem>
#include <filesystem>
#include <iostream>
#include <unordered_map>

#include "inference/hier_parameter_server.hpp"
#include "inference/timer.h"
#include "spdk_wrapper.h"
namespace ssdps {

template <typename KEY_T>
class SsdPsInterface {
 public:
  virtual void BulkLoad(ConstArray<KEY_T> keys_array, const void *value) = 0;
  virtual void BatchGet(ConstArray<KEY_T> keys_array,
                        ConstArray<uint64_t> index, void *dst) = 0;
  virtual ~SsdPsInterface(){};
};

template <typename KEY_T>
class NaiveArraySSD : public SsdPsInterface<KEY_T> {
 public:
  NaiveArraySSD(int VALUE_SIZE, uint64_t vector_capability)
      : VALUE_SIZE(VALUE_SIZE), vector_capability(vector_capability) {
    ssd_ = ssdps::SpdkWrapper::create();
    ssd_->Init();
    bouncedBuffer_ = spdk_malloc(kBouncedBuffer_ * ssd_->GetLBASize(), 0, NULL,
                                 SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    // cudaMallocHost(&bouncedBuffer_, kBouncedBuffer_ * ssd_->GetLBASize(),
    //                cudaHostAllocDefault);
    CHECK(bouncedBuffer_);
  }

  // return <lbaID, InlbaOffset>
  FOLLY_ALWAYS_INLINE
  std::pair<int64_t, int> Mapping(int64_t index) const {
#if 0
    int64_t lba_no = index * VALUE_SIZE / ssd_->GetLBASize();
    int in_lba_offset = (index * VALUE_SIZE) % ssd_->GetLBASize();
    return std::make_pair(lba_no, in_lba_offset);
#endif
#if 0
    int64_t lba_no = index * 1;
    int in_lba_offset = 0;
    return std::make_pair(lba_no, in_lba_offset);
#endif
    uint64_t lba_no = ssd_->GetLBANumber() * index / vector_capability;
    int in_lba_offset = 0;
    return std::make_pair(lba_no, in_lba_offset);
  }

  // the address the index th value stored
  FOLLY_ALWAYS_INLINE
  int64_t MappingLogicalAddress(int64_t index) const {
    int64_t lba_no;
    int in_lba_offset;
    std::tie(lba_no, in_lba_offset) = Mapping(index);
    return lba_no * ssd_->GetLBASize() + in_lba_offset;
  }

  void BulkLoad(ConstArray<KEY_T> keys_array, const void *value) override {
    in_memory_index_.clear();
    in_memory_index_.reserve(keys_array.Size());
    // LOG(ERROR) << "ArraySSD: Load " << ssd_pages << " pages ("
    //            << ssd_pages * ssd_->GetLBASize() / 1024 / 1024 << "MB)";

    const int nr_batch_pages = 32;
    int64_t pinned_bytes = ssd_->GetLBASize() * nr_batch_pages;
    char *pinned_value = (char *)spdk_malloc(
        pinned_bytes, 0, NULL, SPDK_ENV_SOCKET_ID_ANY, SPDK_MALLOC_DMA);
    CHECK(pinned_value) << "spdk_malloc";

    uint64_t start_offset = 0;
    uint64_t batch_start_offset = 0;
    uint64_t batch_end_offset = 0;
    std::vector<uint64_t> indexes_array;
    while (start_offset < keys_array.Size()) {
      FB_LOG_EVERY_MS(INFO, 2000) << fmt::format(
          "SSD load data {} %", 100 * start_offset / keys_array.Size());
      if (1 + Mapping(start_offset).first - Mapping(batch_start_offset).first >
          nr_batch_pages) {
        batch_end_offset = start_offset - 1;
        // 左闭右闭
        // write keys_array[batch_start_offset, batch_end_offset]

        // LOG(INFO) << fmt::format("keys_array[{}, {}]", batch_start_offset,
        //                          batch_end_offset);
        SubBulkLoad(
            nr_batch_pages,
            keys_array.SubArray(batch_start_offset, batch_end_offset + 1),
            indexes_array, (char *)value + VALUE_SIZE * batch_start_offset,
            pinned_value);
        batch_start_offset = start_offset;
        indexes_array.clear();
      }
      indexes_array.push_back(start_offset);
      start_offset++;
    }

    if (batch_start_offset != keys_array.Size()) {
      batch_end_offset = keys_array.Size() - 1;
      SubBulkLoad(nr_batch_pages,
                  keys_array.SubArray(batch_start_offset, batch_end_offset + 1),
                  indexes_array,
                  (char *)value + VALUE_SIZE * batch_start_offset,
                  pinned_value);
    }
    spdk_free(pinned_value);
    // {
    //   void *check_buffer = malloc(keys_array.Size() * VALUE_SIZE);

    //   for (size_t i = 0; i < keys_array.Size(); i += 5000) {
    //     BatchGet(
    //         keys_array.SubArray(i, std::min(int(i + 5000), keys_array.Size())),
    //         ConstArray<uint64_t>(), (char *)check_buffer + i * VALUE_SIZE);
    //   }

    //   for (size_t i = 0; i < keys_array.Size(); i++) {
    //     int ret = memcmp((char *)check_buffer + VALUE_SIZE * i,
    //                      (char *)value + VALUE_SIZE * i, VALUE_SIZE);
    //     if (ret != 0)
    //       LOG(FATAL) << "diff " << i << " , end = " << keys_array.Size();
    //   }
    //   free(check_buffer);
    // }
  }

  static void BulkLoadCB(void *ctx, const struct spdk_nvme_cpl *cpl) {
    if (FOLLY_UNLIKELY(spdk_nvme_cpl_is_error(cpl))) {
      LOG(FATAL) << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }
    std::atomic_int *counter = (std::atomic_int *)ctx;
    counter->fetch_add(1);
  }

  // batch get keys and save to dst with index, the index stores the slot number
  // of dst matrix (i.e. we need * VALUE_SIZE)
  void BatchGet(ConstArray<KEY_T> keys_array, ConstArray<uint64_t> index,
                void *dst) override {
    static std::vector<ReadCompleteCBContext> cb_contexts(kBouncedBuffer_);
    CHECK_LE(keys_array.Size(), kBouncedBuffer_);
    const bool doNotUseIndex = true;
    bool orderedByIndex = true;
    if (index.Data() != nullptr) {
      CHECK_EQ(keys_array.Size(), index.Size());
    } else {
      orderedByIndex = false;
    }

    std::atomic<int> readCompleteCount{0};

    xmh::Timer timer_kvell_index("Hier-SSD index");
    xmh::Timer timer_kvell_submitCommand("Hier-SSD command");
    for (int64_t i = 0; i < keys_array.Size(); i++) {
      int64_t count_offset = -1;
      if (doNotUseIndex) {
        count_offset = keys_array[i];
      } else {
        timer_kvell_index.CumStart();
        auto key = keys_array[i];
        auto iter = in_memory_index_.find(key);
        timer_kvell_index.CumEnd();
        if (iter == in_memory_index_.end()) {
          LOG(FATAL) << "Not find key " << key << " in naiveKvell";
        }
        count_offset = iter->second;
      }
      timer_kvell_submitCommand.CumStart();
      CHECK_LE(VALUE_SIZE, ssd_->GetLBASize()) << "KISS";
      int64_t lba_no;
      int in_lba_offset;
      std::tie(lba_no, in_lba_offset) = Mapping(count_offset);

      auto &ctx = cb_contexts[i];
      ctx.src = (char *)bouncedBuffer_ + i * ssd_->GetLBASize() + in_lba_offset;
      ctx.readCompleteCount = &readCompleteCount;
      if (orderedByIndex)
        ctx.dst = (char *)dst + index[i] * VALUE_SIZE;
      else
        ctx.dst = (char *)dst + i * VALUE_SIZE;
      ctx.value_size = VALUE_SIZE;
      ssd_->SubmitReadCommand((char *)bouncedBuffer_ + i * ssd_->GetLBASize(),
                              VALUE_SIZE, lba_no, ReadCompleteCB, &ctx);
      timer_kvell_submitCommand.CumEnd();
    }
    timer_kvell_index.CumReport();
    timer_kvell_submitCommand.CumReport();

    // batch sync
    xmh::Timer timer_kvell_pollCQ("Hier-SSD PollCQ");
    while (readCompleteCount != keys_array.Size()) {
      ssd_->PollCompleteQueue();
    }
    timer_kvell_pollCQ.end();
  }

  ~NaiveArraySSD() {}

 private:
  // keys_array:  [5,6,7]
  // indexs_array: [5,6,7]
  void SubBulkLoad(const int nr_batch_pages, ConstArray<KEY_T> keys_array,
                   const std::vector<uint64_t> &indexs_array, const void *value,
                   char *pinned_value) {
    // CHECK_EQ(nr_batch_pages * ssd_->GetLBASize(),
    //          keys_array.Size() * VALUE_SIZE);
    CHECK(keys_array.Size() == indexs_array.size());

    int64_t subarray_size = keys_array.Size();
    for (int64_t i = 0; i < subarray_size; i++) {
      in_memory_index_[keys_array[i]] = indexs_array[i];
    }

    int64_t first_page_lba = Mapping(indexs_array.front()).first;
    int64_t last_page_lba = Mapping(indexs_array.back()).first;

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
              ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter);
          ssd_->PollCompleteQueue();
        } while (ret != 0);
        submit_counter++;
      }
      memcpy(pinned_value + submit_counter * ssd_->GetLBASize() +
                 Mapping(index).second,
             (char *)value + i * VALUE_SIZE, VALUE_SIZE);
      old_page_id = Mapping(index).first;
    }
    // write the last page
    int ret;
    do {
      ret = ssd_->SubmitWriteCommand(
          pinned_value + submit_counter * ssd_->GetLBASize(),
          ssd_->GetLBASize(), old_page_id, BulkLoadCB, &finished_counter);
      ssd_->PollCompleteQueue();
    } while (ret != 0);
    submit_counter++;
    CHECK_LE(submit_counter, last_page_lba - first_page_lba + 1);
    while (submit_counter != finished_counter) ssd_->PollCompleteQueue();
  }

  struct ReadCompleteCBContext {
    void *src;
    void *dst;
    int value_size;
    std::atomic<int> *readCompleteCount;
  };

  // copy VALUE_SIZE bytes from <src> to <dst>
  static void ReadCompleteCB(void *ctx, const struct spdk_nvme_cpl *cpl) {
    ReadCompleteCBContext *readCompleteCBContext = (ReadCompleteCBContext *)ctx;
    if (FOLLY_UNLIKELY(spdk_nvme_cpl_is_error(cpl))) {
      LOG(FATAL) << "I/O error status: "
                 << spdk_nvme_cpl_get_status_string(&cpl->status);
    }
    xmh::Timer ssd_memory_bounce("Hier-SSD memory_bounce");
    memcpy(readCompleteCBContext->dst, readCompleteCBContext->src,
           readCompleteCBContext->value_size);
    readCompleteCBContext->readCompleteCount->fetch_add(1);
    ssd_memory_bounce.end();
  }

  int VALUE_SIZE;
  uint64_t vector_capability;
  std::unordered_map<KEY_T, uint64_t> in_memory_index_;
  static constexpr int kBouncedBuffer_ = 20000;
  void *bouncedBuffer_;
  std::unique_ptr<ssdps::SpdkWrapper> ssd_;
};
}  // namespace ssdps
