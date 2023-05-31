#include <buffer.h>
#include <ctrl.h>
#include <cuda.h>
#include <event.h>
#include <fcntl.h>
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

#include "BaMWrapper.h"
#include "base/cuda.h"

using error = std::runtime_error;
using std::string;

const char *const ctrls_paths[] = {
    "/dev/libnvm0", "/dev/libnvm1", "/dev/libnvm2", "/dev/libnvm3",
    "/dev/libnvm4", "/dev/libnvm5", "/dev/libnvm6", "/dev/libnvm7"};

// clang-format off
__global__ void gdssd_copy_emb_kernel(array_d_t<float> *dr, 
                                const uint64_t* d_index,
                                const int lens,
                                const int dim,
                                float * output_emb) {
// clang-format on 
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int key_index = tid / dim;
  int emb_index = tid % dim;
  if (key_index  < lens) {
      uint64_t dst_float_index = d_index[key_index] *dim + emb_index;
      output_emb[key_index * dim + emb_index] =  (*dr)[dst_float_index];
  }
}


// clang-format off
__global__ void gdssd_init_db_kernel(array_d_t<float> *dr, 
                                const int lens,
                                const int dim
                                 ) {
// clang-format on 
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int key_index = tid / dim;
  int emb_index = tid % dim;
  if (key_index  < lens) {
      int dst_float_index = key_index * dim + emb_index;
      dr->seq_write(dst_float_index, (float)key_index);
  }
}

__global__ void gdssd_check_db_kernel(array_d_t<float> *dr, 
                                const int lens,
                                const int dim
                                 ) {
// clang-format on 
  uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  int key_index = tid / dim;
  int emb_index = tid % dim;
  if (key_index  < lens) {
      int dst_float_index = key_index * dim + emb_index;
      float read_value = (*dr)[dst_float_index] ;
      float truth_value = key_index;
      if(fabs(read_value - truth_value) > 1e-6){
        printf("gdssd_check_db_kernel failed\n");
        assert(0);
        *(int*)0 = 0;
      }
      else{
        // printf("gdssd_check_db_kernel pass\n");
        // assert(0);
        // *(int*)0 = 0;
      }
  }
}



namespace gpu_direct_ssd {

template<typename KEY_T>
  BaMWrapper<KEY_T>::BaMWrapper(uint64_t DBCapacity, int embeddingDimension): kDBCapacity(DBCapacity), kEmbeddingDimension(embeddingDimension){
    const int n_ctrls = 1;
    int cudaDevice = 0;
    const int nvmNamespace = 1;
    const int queueDepth = 1024;
    const int numQueues = 128;

    const int pageSize = this->GetLBASize();
    const int numPages = this->GetPageCacheNumber();

    const uint64_t n_elems = kDBCapacity * kEmbeddingDimension;

    LOG(INFO) << "Init GPU-SSD with DBCapacity = " << (float)DBCapacity/1e6 << " M";
    
    cudaDeviceProp properties;
    if (cudaGetDeviceProperties(&properties, cudaDevice) != cudaSuccess) {
      LOG(FATAL) << "Failed to get CUDA device properties " << cudaDevice;
    }
    std::string card_name(properties.name);
    if (card_name.find("A30") == std::string::npos)
      LOG(FATAL) << "RawGDWrapper using " << properties.name;
   
    for (size_t i = 0; i < n_ctrls; i++) {
      std::cout << "before construct ctrls[i] " << i << std::endl << std::flush;
      ctrls_.push_back(new Controller(ctrls_paths[i], nvmNamespace, cudaDevice, queueDepth,
                                     numQueues));
      std::cout << "after construct ctrls[i] " << i << std::endl << std::flush;
    }

    uint64_t page_size = pageSize;
    uint64_t n_pages = numPages;
    uint64_t total_cache_size = (page_size * n_pages);

    page_cache_t *h_pc = new page_cache_t (page_size, n_pages, cudaDevice, ctrls_[0][0], (uint64_t)64, ctrls_);
    std::cout << "finished creating cache\n";

    // page_cache_t *d_pc = (page_cache_t *)(h_pc.d_pc_ptr);
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
  //(const uint64_t num_elems, const uint64_t disk_start_offset, const
  // std::vector<range_t<T>*>& ranges, Settings& settings)
  ssd_array_ = new array_t<float>(n_elems, 0, vr, cudaDevice);
  std::cout << "finished creating range\n";
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename KEY_T>
void BaMWrapper<KEY_T>::InitFakeDB() {
  uint64_t b_size = 256;
  uint64_t numThreads = kDBCapacity * kEmbeddingDimension;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;  // 80*16;
  LOG(INFO) << "launch gdssd_init_db_kernel with (" << g_size << ", " << b_size
            << ")";
  gdssd_init_db_kernel<<<g_size, b_size>>>(ssd_array_->d_array_ptr, kDBCapacity,
                                           kEmbeddingDimension);
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
  LOG(INFO) << "launch gdssd_check_db_kernel with (" << g_size << ", " << b_size
            << ")";
  gdssd_check_db_kernel<<<g_size, b_size>>>(ssd_array_->d_array_ptr,
                                            kDBCapacity, kEmbeddingDimension);
  XMH_CUDA_CHECK(cudaGetLastError());
  XMH_CUDA_CHECK(cudaDeviceSynchronize());
}

template <typename KEY_T>
void BaMWrapper<KEY_T>::BulkLoad(base::ConstArray<KEY_T> keys_array,
                                 const void *value) {
  CHECK_EQ(keys_array.Size(), kDBCapacity);
}

template <typename KEY_T>
void BaMWrapper<KEY_T>::Write(void *h_input, uint64_t startBlock,
                              int block_num) {}

// d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
template <typename KEY_T>
double BaMWrapper<KEY_T>::Query(float *d_outputBuffer, const KEY_T *d_index,
                                const int count, cudaStream_t stream) {
  uint64_t b_size = 128;
  uint64_t numThreads = count * kEmbeddingDimension;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;  // 80*16;
  gdssd_copy_emb_kernel<<<g_size, b_size, 0, stream>>>(
      ssd_array_->d_array_ptr, (const uint64_t *)d_index, count,
      kEmbeddingDimension, d_outputBuffer);
  XMH_CUDA_CHECK(cudaGetLastError());
  return 0;
}

template <typename KEY_T>
void BaMWrapper<KEY_T>::PrintResetStats() {
  ssd_array_->print_reset_stats();
}

template class BaMWrapper<unsigned int>;
template class BaMWrapper<long long>;
template class BaMWrapper<uint64_t>;

}  // namespace gpu_direct_ssd