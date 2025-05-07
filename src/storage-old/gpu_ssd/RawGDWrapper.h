#pragma once
#include "base/array.h"
#include "interface.h"

class Controller;

class page_cache_t;
class page_cache_d_t;

namespace gpu_direct_ssd {
template <typename KEY_T>
class RawGDWrapper : public GPUDirectSSDInterface<KEY_T> {
 public:
  RawGDWrapper(uint64_t DBCapacity, int embeddingDimension);

  void InitFakeDB();

  void BulkLoad(base::ConstArray<KEY_T> keys_array, const void *value);

  void Write(void *h_input, uint64_t startBlock, int block_num);

  // d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
  double Query(float *d_outputBuffer, const KEY_T *d_index, const int count,
               cudaStream_t stream);

  static void outputFile(void *d_data, size_t size, const char *filename);

  void WriteBlock(void *h_buffer, int lba_no, int nr_lba);

  void PrintResetStats();

  virtual ~RawGDWrapper() {}

 private:
  void SubBulkLoad(const int nr_batch_pages, base::ConstArray<KEY_T> keys_array,
                   const std::vector<uint64_t> &indexs_array, const void *value,
                   char *pinned_value);
  std::vector<Controller *> ctrls_;
  page_cache_t *h_pc;
  page_cache_d_t *d_pc;
  const uint64_t kDBCapacity;
  const int kEmbeddingDimension;
};
}  // namespace gpu_direct_ssd