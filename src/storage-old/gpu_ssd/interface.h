#pragma once
#include "base/array.h"

namespace gpu_direct_ssd {
template <typename KEY_T>
class GPUDirectSSDInterface {
 public:
  virtual void InitFakeDB() = 0;
  virtual void BulkLoad(base::ConstArray<KEY_T> keys_array, const void *value) = 0;
  virtual void Write(void *h_input, uint64_t startBlock, int block_num) = 0;

  // d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
  virtual double Query(float *d_outputBuffer, const KEY_T *d_index,
                       const int count, cudaStream_t stream) = 0;

  virtual int GetLBASize() const { return 512; };

  virtual int GetPageCacheNumber() const { return 102400; };

  virtual ~GPUDirectSSDInterface() {}
};
}  // namespace gpu_direct_ssd