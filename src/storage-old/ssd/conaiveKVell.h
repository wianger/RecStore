#pragma once
#include <folly/GLog.h>
#include <folly/Portability.h>
#include <folly/system/MemoryMapping.h>
#include <boost/coroutine2/all.hpp>

#include <boost/algorithm/string/join.hpp>
#include <experimental/filesystem>
#include <filesystem>
#include <iostream>
#include <unordered_map>

#include "base/array.h"
#include "base/timer.h"
#include "spdk_wrapper.h"
namespace ssdps {

using base::ConstArray;
using boost::coroutines2::coroutine;

template <typename KEY_T>
class CoNaiveArraySSD {
 public:
  CoNaiveArraySSD(int VALUE_SIZE, uint64_t vector_capability, int thread_num)
      : VALUE_SIZE(VALUE_SIZE), vector_capability(vector_capability), thread_num(thread_num) {
    CHECK(thread_num <= MAX_QUEUE_NUM);
    ssd_ = ssdps::SpdkWrapper::create(thread_num);
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

  // return <lbaID, InlbaOffset>
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

  // the address the index th value stored
  FOLLY_ALWAYS_INLINE
  int64_t MappingLogicalAddress(int64_t index) const {
    int64_t lba_no;
    int in_lba_offset;
    std::tie(lba_no, in_lba_offset) = Mapping(index);
    return lba_no * ssd_->GetLBASize() + in_lba_offset;
  }

  void BatchPut(ConstArray<uint64_t> keys_array, const void *value, int tid) {
    const int nr_batch_pages = 32;
    int i = 0;
    while(i < keys_array.Size()) {
      int batched_size = std::min(nr_batch_pages, keys_array.Size() - i);
      SubBulkLoad(
          batched_size,
          keys_array.SubArray(i, i + batched_size), (char *)value + VALUE_SIZE * i,
          write_buffer_[tid], tid);
      i += batched_size;
    }
  }

  void BatchPut(ConstArray<uint64_t> keys_array, std::vector<ConstArray<float>> &value, int tid) {
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

  void BulkLoad(int keys_size, const void *value) {
    // LOG(ERROR) << "ArraySSD: Load " << ssd_pages << " pages ("
    //            << ssd_pages * ssd_->GetLBASize() / 1024 / 1024 << "MB)";
    std::vector<uint64_t> keys_vec;
    for(int i = 0; i < keys_size; i++) {
      keys_vec.push_back(i);
    }
    BatchPut(ConstArray<uint64_t>(keys_vec), value, 0);
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
  void BatchGet(coroutine<void>::push_type& sink, ConstArray<KEY_T> keys_array, ConstArray<uint64_t> index,
                void *dst, int tid) {
    static thread_local std::vector<ReadCompleteCBContext> cb_contexts(kBouncedBuffer_);
    CHECK_LE(keys_array.Size(), kBouncedBuffer_);
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
      count_offset = keys_array[i];
      timer_kvell_submitCommand.CumStart();
      CHECK_LE(VALUE_SIZE, ssd_->GetLBASize()) << "KISS";
      int64_t lba_no;
      int in_lba_offset;
      std::tie(lba_no, in_lba_offset) = Mapping(count_offset);

      auto &ctx = cb_contexts[i];
      ctx.src = bouncedBuffer_[tid] + i * ssd_->GetLBASize() + in_lba_offset;
      ctx.readCompleteCount = &readCompleteCount;
      if (orderedByIndex)
        ctx.dst = (char *)dst + index[i] * VALUE_SIZE;
      else
        ctx.dst = (char *)dst + i * VALUE_SIZE;
      ctx.value_size = VALUE_SIZE;
      ssd_->SubmitReadCommand(bouncedBuffer_[tid] + i * ssd_->GetLBASize(),
                              VALUE_SIZE, lba_no, ReadCompleteCB, &ctx, tid);
      timer_kvell_submitCommand.CumEnd();
    }
    timer_kvell_index.CumReport();
    timer_kvell_submitCommand.CumReport();
    sink();
    // batch sync
    xmh::Timer timer_kvell_pollCQ("Hier-SSD PollCQ");
    while (readCompleteCount != keys_array.Size()) {
      ssd_->PollCompleteQueue(tid);
    }
    timer_kvell_pollCQ.end();
  }

  ~CoNaiveArraySSD() {
    spdk_free(rawwrite_buffer_);
    spdk_free(rawbouncedBuffer_);
  }

 private:
  // keys_array:  [5,6,7]
  // indexs_array: [5,6,7]
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
             (char *)value + i * VALUE_SIZE, VALUE_SIZE);
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

  void SubBulkLoad(int keys_size, base::ConstArray<uint64_t> indexs_array, std::vector<ConstArray<float>> &value, int start,
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
             value[i + start].Data(), VALUE_SIZE);
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
    memcpy(readCompleteCBContext->dst, readCompleteCBContext->src,
           readCompleteCBContext->value_size);
    readCompleteCBContext->readCompleteCount->fetch_add(1);
  }
  static const int MAX_QUEUE_NUM = 32;
  int VALUE_SIZE;
  uint64_t vector_capability;
  static constexpr int kBouncedBuffer_ = 20000;
  char *rawbouncedBuffer_;
  char *rawwrite_buffer_;
  char *bouncedBuffer_[MAX_QUEUE_NUM];
  char *write_buffer_[MAX_QUEUE_NUM];
  std::unique_ptr<ssdps::SpdkWrapper> ssd_;
  int thread_num;
};
}  // namespace ssdps
