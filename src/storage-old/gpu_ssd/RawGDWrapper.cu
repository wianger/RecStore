#include <buffer.h>
#include <ctrl.h>
#include <cuda.h>
#include <event.h>
#include <fcntl.h>
#include <fmt/core.h>
#include <nvm_admin.h>
#include <nvm_cmd.h>
#include <nvm_ctrl.h>
#include <nvm_error.h>
#include <nvm_io.h>
#include <nvm_parallel_queue.h>
#include <nvm_queue.h>
#include <nvm_types.h>
#include <nvm_util.h>
#include <page_cache.h>
#include <queue.h>
#include <sys/mman.h>
#include <unistd.h>
#include <util.h>

#include "folly/GLog.h"
// #include "folly/Format.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "settings.h"
#ifdef __DIS_CLUSTER__
#include <sisci_api.h>
#endif

#include <folly/GLog.h>
// #include <folly/Format.h>

#include "RawGDWrapper.h"
#include "base/cuda.h"

using error = std::runtime_error;
using std::string;

const char *const ctrls_paths[] = {
    "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3",
    "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7"};

__forceinline__ __device__ uint64_t xmh_get_cache_page_addr_device(
    page_cache_d_t *cache, const uint32_t page_trans) {
  return ((uint64_t)((cache->base_addr + (page_trans * cache->page_size))));
}

__forceinline__ uint64_t
xmh_get_cache_page_addr_host(page_cache_t *cache, const uint32_t page_trans) {
  return (
      (uint64_t)((cache->pdt.base_addr + (page_trans * cache->pdt.page_size))));
}

inline __device__ uint16_t xmh_sq_enqueue(
    nvm_queue_t *sq, nvm_cmd_t *cmd,
    simt::atomic<uint64_t, simt::thread_scope_device> *pc_tail = NULL,
    uint64_t *cur_pc_tail = NULL) {
  uint32_t ticket;
  ticket = sq->in_ticket.fetch_add(1, simt::memory_order_relaxed);
  uint32_t pos = ticket & (sq->qs_minus_1);
  uint64_t id = get_id(ticket, sq->qs_log2);

  unsigned int ns = 8;
  while ((sq->tickets[pos].val.load(simt::memory_order_relaxed) != id)) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
    __nanosleep(ns);
    if (ns < 256) {
      ns *= 2;
    }
#endif
  }

  ns = 8;
  while ((sq->tickets[pos].val.load(simt::memory_order_acquire) != id)) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
    __nanosleep(ns);
    if (ns < 256) {
      ns *= 2;
    }
#endif
  }

  copy_type *queue_loc = ((copy_type *)(((nvm_cmd_t *)(sq->vaddr)) + pos));
  copy_type *cmd_ = ((copy_type *)(cmd->dword));
  // printf("+++tid: %llu\tcid: %llu\tsq_loc: %llx\tpos: %llu\n", (unsigned long
  // long) (threadIdx.x+blockIdx.x*blockDim.x), (unsigned long long)
  // (cmd->dword[0] >> 16), (unsigned long long) queue_loc, (unsigned long long)
  // pos);

#pragma unroll
  for (uint32_t i = 0; i < 64 / sizeof(copy_type); i++) {
    queue_loc[i] = cmd_[i];
  }

  if (pc_tail) {
    *cur_pc_tail = pc_tail->load(simt::memory_order_relaxed);
  }
  sq->tail_mark[pos].val.store(LOCKED, simt::memory_order_release);

  bool cont = true;
  ns = 8;
  cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
  while (cont) {
    bool new_cont = sq->tail_lock.load(simt::memory_order_relaxed) == LOCKED;
    if (!new_cont) {
      new_cont =
          sq->tail_lock.fetch_or(LOCKED, simt::memory_order_acquire) == LOCKED;
      if (!new_cont) {
        uint32_t cur_tail = sq->tail.load(simt::memory_order_relaxed);

        uint32_t tail_move_count = move_tail(sq, cur_tail);

        if (tail_move_count) {
          uint32_t new_tail = cur_tail + tail_move_count;
          uint32_t new_db = (new_tail) & (sq->qs_minus_1);
          if (pc_tail) {
            *cur_pc_tail = pc_tail->load(simt::memory_order_acquire);
          }
          *(sq->db) = new_db;

          // sq->tail_copy.store(new_tail, simt::memory_order_release);
          //	            printf("wrote SQ_db: %llu\tcur_tail:
          //%llu\tmove_count: %llu\tsq_tail: %llu\tsq_head: %llu\n", (unsigned
          // long long) new_db, (unsigned
          // long long) cur_tail, (unsigned long long) tail_move_count,
          // (unsigned long long) (new_tail),  (unsigned long
          // long)(sq->head.load(simt::memory_order_acquire)));
          sq->tail.store(new_tail, simt::memory_order_release);
          // cont = false;
        }
        sq->tail_lock.store(UNLOCKED, simt::memory_order_release);
      }
    }
    cont = sq->tail_mark[pos].val.load(simt::memory_order_relaxed) == LOCKED;
    if (cont) {
#if defined(__CUDACC__) && (__CUDA_ARCH__ >= 700 || !defined(__CUDA_ARCH__))
      __nanosleep(ns);
      if (ns < 256) {
        ns *= 2;
      }
#endif
    }
  }

  sq->tickets[pos].val.fetch_add(1, simt::memory_order_acq_rel);
  return pos;
}

inline __device__ void xmh_read_data(page_cache_d_t *pc, QueuePair *qp,
                                     const uint64_t starting_lba,
                                     const uint64_t n_blocks,
                                     const unsigned long long pc_entry) {
  //   printf("read LBA %lld+%lld to %lld\n", starting_lba, n_blocks, pc_entry);
  nvm_cmd_t cmd;
  uint16_t cid = get_cid(&(qp->sq));
  nvm_cmd_header(&cmd, cid, NVM_IO_READ, qp->nvmNamespace);
  uint64_t prp1 = pc->prp1[pc_entry];
  uint64_t prp2 = 0;
  if (pc->prps) prp2 = pc->prp2[pc_entry];
  nvm_cmd_data_ptr(&cmd, prp1, prp2);
  nvm_cmd_rw_blks(&cmd, starting_lba, n_blocks);
  uint16_t sq_pos = xmh_sq_enqueue(&qp->sq, &cmd);
  uint32_t head, head_;

  uint32_t cq_pos = cq_poll(&qp->cq, cid, &head, &head_);

  qp->cq.tail.fetch_add(1, simt::memory_order_acq_rel);
  uint64_t pc_prev_head = pc->q_head->load(simt::memory_order_relaxed);
  uint64_t pc_pos = pc->q_tail->fetch_add(1, simt::memory_order_acq_rel);

  cq_dequeue(&qp->cq, cq_pos, &qp->sq, head, head_);

  // enqueue_second(page_cache_d_t* pc, QueuePair* qp, const uint64_t
  // starting_lba, nvm_cmd_t* cmd, const uint16_t cid, const uint64_t pc_pos,
  // const uint64_t pc_prev_head)
  enqueue_second(pc, qp, starting_lba, &cmd, cid, pc_pos, pc_prev_head);

  put_cid(&qp->sq, cid);
}

template <typename KEY_T>
__global__ void fused_batch_read_kernel(Controller **ctrls, page_cache_d_t *pc,
                                        const KEY_T *d_keys,
                                        const int nr_requests,
                                        const int nr_lbas_per_thread,
                                        const int value_size, float *d_dst) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t key_id = tid / (value_size / sizeof(float));
  uint32_t laneid = lane_id();
  uint32_t bid = blockIdx.x;
  uint32_t smid = get_smid();

  uint32_t ctrl;
  uint32_t queue;
  if (laneid == 0) {
    ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) %
           (pc->n_ctrls);
    queue = smid % (ctrls[ctrl]->n_qps);
  }
  ctrl = __shfl_sync(0xFFFFFFFF, ctrl, 0);
  queue = __shfl_sync(0xFFFFFFFF, queue, 0);

  if (key_id < nr_requests) {
    uint64_t start_lba = d_keys[key_id];
    uint64_t in_lba_offset = 0;

    uint64_t page_cache_slot = key_id * nr_lbas_per_thread;
    assert(page_cache_slot < pc->n_pages);

    if (key_id * (value_size / sizeof(float)) == tid)
      xmh_read_data(pc, (ctrls[ctrl]->d_qps) + (queue), start_lba,
                    nr_lbas_per_thread, page_cache_slot);

    warp_memcpy<float>(
        &d_dst[value_size / sizeof(float) * key_id],
        (float *)(xmh_get_cache_page_addr_device(pc, page_cache_slot) +
                  in_lba_offset),
        value_size / sizeof(float));
  }
}

template <typename KEY_T>
__global__ void batch_submit_read_kernel(Controller **ctrls, page_cache_d_t *pc,
                                         const KEY_T *d_keys,
                                         const int nr_requests,
                                         const int nr_lbas_per_thread) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t laneid = lane_id();
  uint32_t bid = blockIdx.x;
  uint32_t smid = get_smid();

  uint32_t ctrl;
  uint32_t queue;
  if (laneid == 0) {
    ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) %
           (pc->n_ctrls);
    queue = smid % (ctrls[ctrl]->n_qps);
  }
  ctrl = __shfl_sync(0xFFFFFFFF, ctrl, 0);
  queue = __shfl_sync(0xFFFFFFFF, queue, 0);

  if (tid < nr_requests) {
    uint64_t start_lba = d_keys[tid];
    uint64_t in_lba_offset = 0;

    uint64_t page_cache_slot = tid * nr_lbas_per_thread;
    assert(page_cache_slot < pc->n_pages);
    xmh_read_data(pc, (ctrls[ctrl]->d_qps) + (queue), start_lba,
                  nr_lbas_per_thread, page_cache_slot);
  }
}

template <typename KEY_T>
__global__ void batch_read_pagecache_kernel(
    Controller **ctrls, page_cache_d_t *pc, const KEY_T *d_keys,
    const int nr_requests, const int nr_lbas_per_thread, const int value_size,
    float *d_dst) {
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint64_t key_id = tid / 32;

  uint32_t laneid = lane_id();
  uint32_t bid = blockIdx.x;
  uint32_t smid = get_smid();

  uint32_t ctrl;
  uint32_t queue;
  if (laneid == 0) {
    ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) %
           (pc->n_ctrls);
    queue = smid % (ctrls[ctrl]->n_qps);
  }
  ctrl = __shfl_sync(0xFFFFFFFF, ctrl, 0);

  if (key_id < nr_requests) {
    uint64_t start_lba = d_keys[key_id];
    uint64_t in_lba_offset = 0;

    uint64_t page_cache_slot = key_id * nr_lbas_per_thread;
    assert(page_cache_slot < pc->n_pages);
    // copy page_cache_slot + in_lba_offset, size=value_size
    // to d_dst[value_size*tid], size=value_size
    warp_memcpy<float>(
        &d_dst[value_size / sizeof(float) * key_id],
        (float *)(xmh_get_cache_page_addr_device(pc, page_cache_slot) +
                  in_lba_offset),
        value_size / sizeof(float));
  }
}

__global__ void sequential_write_kernel(Controller **ctrls, page_cache_d_t *pc,
                                        uint64_t start_lba, uint32_t nr_lbas) {
  assert(nr_lbas == 1), "now we only support once write a page";
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t laneid = lane_id();
  uint32_t bid = blockIdx.x;
  uint32_t smid = get_smid();

  uint32_t ctrl;
  uint32_t queue;
  if (laneid == 0) {
    ctrl = pc->ctrl_counter->fetch_add(1, simt::memory_order_relaxed) %
           (pc->n_ctrls);
    queue = smid % (ctrls[ctrl]->n_qps);
  }
  ctrl = __shfl_sync(0xFFFFFFFF, ctrl, 0);
  queue = __shfl_sync(0xFFFFFFFF, queue, 0);

  if (tid < 1) {
    uint64_t page_cache_slot = 0;
    write_data(pc, (ctrls[ctrl]->d_qps) + (queue), start_lba, nr_lbas,
               page_cache_slot);
  }
}

namespace gpu_direct_ssd {

template <typename KEY_T>
RawGDWrapper<KEY_T>::RawGDWrapper(uint64_t DBCapacity, int embeddingDimension)
    : kDBCapacity(DBCapacity), kEmbeddingDimension(embeddingDimension) {
  const int n_ctrls = 1;
  int cudaDevice = -1;
  const int nvmNamespace = 1;
  const int queueDepth = 1024;
  const int numQueues = 128;
  //   const int numQueues = 1;

  const int pageSize = this->GetLBASize();
  const int numPages = this->GetPageCacheNumber();

  const uint64_t n_elems = kDBCapacity * kEmbeddingDimension;

  LOG(INFO) << fmt::format("Init GPU-SSD with DBCapacity={}M, Dimension={}",
                           (float)DBCapacity / 1e6, kEmbeddingDimension);
  if (cudaGetDevice((int *)&cudaDevice) != cudaSuccess) {
    LOG(FATAL) << "Failed to cudaGetDevice";
  }
  cudaDeviceProp properties;
  if (cudaGetDeviceProperties(&properties, cudaDevice) != cudaSuccess) {
    LOG(FATAL) << "Failed to get CUDA device properties " << cudaDevice;
  }
  std::string card_name(properties.name);
  if (card_name.find("A30") == std::string::npos)
    LOG(FATAL) << "RawGDWrapper using " << properties.name;

  LOG(INFO) << "RawGDWrapper using " << properties.name;
  // settings
  for (size_t i = 0; i < n_ctrls; i++) {
    std::cout << "before construct ctrls[i] " << i << std::endl << std::flush;
    ctrls_.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice,
                                    queueDepth, numQueues));
    std::cout << "after construct ctrls[i] " << i << std::endl << std::flush;
  }

  uint64_t page_size = pageSize;
  uint64_t n_pages = numPages;
  uint64_t total_cache_size = (page_size * n_pages);

  h_pc = new page_cache_t(page_size, n_pages, cudaDevice, ctrls_[0][0],
                          (uint64_t)64, ctrls_);
  d_pc = h_pc->d_pc_ptr;
  std::cout << "finished creating cache\n";

  uint64_t ssd_data_size = n_elems * sizeof(float);

  // clang-format off
    range_t<float> *h_range = new range_t<float>(
      (uint64_t)0,           // index start, 逻辑上这个range涵盖的下标开始
      (uint64_t)n_elems,     // count,       逻辑上这个range涵盖的下标数
      (uint64_t)0,           //page_start    物理上页面开始
      (uint64_t)((ssd_data_size+page_size-1)/page_size),  // page_count
      (uint64_t)0,           // page_start_offset
      (uint64_t)page_size,   // page_size
      h_pc,                 // page_cache
      cudaDevice);
  // clang-format on

  range_t<float> *d_range = (range_t<float> *)h_range->d_range_ptr;
  std::vector<range_t<float> *> vr(1);
  vr[0] = h_range;
  std::cout << "finished creating range\n";
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename KEY_T>
static std::pair<int, int> Mapping(KEY_T key) {
  return std::make_pair(key, 0);
}

template <typename KEY_T>
void RawGDWrapper<KEY_T>::WriteBlock(void *h_value, int lba_no, int nr_lba) {
  CHECK_EQ(nr_lba, 1) << "we only support write a page a time";
  const uint64_t pc_addr = xmh_get_cache_page_addr_host(h_pc, 0);
  cudaMemcpy((void *)pc_addr, h_value, 512, cudaMemcpyDefault);
  CHECK_LE(nr_lba, h_pc->pdt.n_pages);
  XMH_CUDA_CHECK(cudaGetLastError());
  sequential_write_kernel<<<1, 32>>>(h_pc->pdt.d_ctrls, d_pc, lba_no, nr_lba);
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
}

// keys_array:  [5,6,7]
// indexs_array: [5,6,7]
template <typename KEY_T>
void RawGDWrapper<KEY_T>::SubBulkLoad(const int nr_batch_pages,
  base::ConstArray<KEY_T> keys_array,
                                      const std::vector<uint64_t> &indexs_array,
                                      const void *value, char *pinned_value) {
  //   static int isFirstVisit = 0;
  //   if (isFirstVisit >= 100) return;
  //   isFirstVisit++;

  int page_cache_size = h_pc->pdt.page_size * nr_batch_pages;

  const int VALUE_SIZE = kEmbeddingDimension * sizeof(float);
  const uint64_t pc_addr = xmh_get_cache_page_addr_host(h_pc, 0);
  CHECK(keys_array.Size() == indexs_array.size());
  int64_t first_page_lba = Mapping(indexs_array.front()).first;
  int64_t last_page_lba = Mapping(indexs_array.back()).first;

  //   std::cout << fmt::format("SubBulkLoad [{}, {}]\n", first_page_lba,
  //   last_page_lba);

  int nr_lbas = last_page_lba - first_page_lba + 1;

  CHECK_LE(nr_lbas, nr_batch_pages);

  memset(pinned_value, 0, page_cache_size);

  for (int i = 0; i < keys_array.Size(); i++) {
    CHECK_EQ(indexs_array[i], keys_array[i]);
    int64_t lba_no = Mapping(indexs_array[i]).first;
    int64_t lba_offset = Mapping(indexs_array[i]).second;

    memcpy(pinned_value + (lba_no - first_page_lba) * this->GetLBASize() +
               lba_offset,
           (char *)value + i * VALUE_SIZE, VALUE_SIZE);
  }
  cudaMemcpy((void *)pc_addr, pinned_value, page_cache_size, cudaMemcpyDefault);
  XMH_CUDA_CHECK(cudaGetLastError());
  sequential_write_kernel<<<1, 32>>>(h_pc->pdt.d_ctrls, d_pc, first_page_lba,
                                     nr_lbas);
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename KEY_T>
void RawGDWrapper<KEY_T>::InitFakeDB() {
  std::vector<KEY_T> key_vec;
  std::vector<float> value_vec;
  key_vec.reserve(kDBCapacity);
  value_vec.reserve(kDBCapacity * kEmbeddingDimension);
  for (uint64_t i = 0; i < kDBCapacity; i++) {
    key_vec.push_back(i);
    for (int j = 0; j < kEmbeddingDimension; j++) {
      value_vec.push_back(i);
    }
  }
  BulkLoad(base::ConstArray<KEY_T>(key_vec), value_vec.data());
}

template <typename KEY_T>
void RawGDWrapper<KEY_T>::BulkLoad(base::ConstArray<KEY_T> keys_array,
                                   const void *value) {
  CHECK_EQ(keys_array.Size(), kDBCapacity);
  //   const int nr_batch_pages = h_pc->pdt.n_pages;
  const int nr_batch_pages = 1;

  char *pinned_value;
  int page_cache_size = h_pc->pdt.page_size * nr_batch_pages;
  cudaMallocHost(&pinned_value, page_cache_size);

  const int VALUE_SIZE = kEmbeddingDimension * sizeof(float);

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

      SubBulkLoad(nr_batch_pages,
                  keys_array.SubArray(batch_start_offset, batch_end_offset + 1),
                  indexes_array,
                  (char *)value + VALUE_SIZE * batch_start_offset,
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
                indexes_array, (char *)value + VALUE_SIZE * batch_start_offset,
                pinned_value);
  }

  cudaFreeHost(pinned_value);
}

template <typename KEY_T>
void RawGDWrapper<KEY_T>::Write(void *h_input, uint64_t startBlock,
                                int block_num) {}

// d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
template <typename KEY_T>
double RawGDWrapper<KEY_T>::Query(float *d_outputBuffer, const KEY_T *d_index,
                                  const int count, cudaStream_t stream) {
#if 1
  uint64_t b_size = 128;
  uint64_t numThreads = count * kEmbeddingDimension;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  int nr_lbas_per_kv =
      (this->GetLBASize() - 1 + kEmbeddingDimension * sizeof(float)) /
      this->GetLBASize();

  fused_batch_read_kernel<KEY_T><<<g_size, b_size, 0, stream>>>(
      h_pc->pdt.d_ctrls, d_pc, d_index, count, nr_lbas_per_kv,
      kEmbeddingDimension * sizeof(float), d_outputBuffer);
  return 0;

#else
  uint64_t b_size = 128;
  uint64_t numThreads = count;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;

  int nr_lbas_per_kv =
      (GetLBASize() - 1 + kEmbeddingDimension * sizeof(float)) / GetLBASize();

  CHECK_EQ(nr_lbas_per_kv, 1);

  batch_submit_read_kernel<KEY_T><<<g_size, b_size, 0, stream>>>(
      h_pc->pdt.d_ctrls, d_pc, d_index, count, nr_lbas_per_kv);
  XMH_CUDA_CHECK(cudaGetLastError());

  numThreads = count * kEmbeddingDimension;
  g_size = (numThreads + b_size - 1) / b_size;
  batch_read_pagecache_kernel<KEY_T><<<g_size, b_size, 0, stream>>>(
      h_pc->pdt.d_ctrls, d_pc, d_index, count, nr_lbas_per_kv,
      kEmbeddingDimension * sizeof(float), d_outputBuffer);
  XMH_CUDA_CHECK(cudaGetLastError());

  return 0;
#endif
}

static void outputFile(void *d_data, size_t size, const char *filename) {
  auto buffer = createBuffer(size);

  cudaError_t err =
      cudaMemcpy(buffer.get(), d_data, size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    throw error(string("Failed to copy data from destination: ") +
                cudaGetErrorString(err));
  }

  FILE *fp = fopen(filename, "wb");
  fwrite(buffer.get(), 1, size, fp);
  fclose(fp);
}

template <typename KEY_T>
void RawGDWrapper<KEY_T>::PrintResetStats() {}

template class RawGDWrapper<unsigned int>;
template class RawGDWrapper<long long>;
template class RawGDWrapper<uint64_t>;

}  // namespace gpu_direct_ssd