#pragma once
#include "base/array.h"
#include "interface.h"

class Controller;

template <typename T>
class array_t;

namespace gpu_direct_ssd {

template <typename KEY_T>
class BaMWrapper : public GPUDirectSSDInterface<KEY_T> {
 public:
  BaMWrapper(uint64_t DBCapacity, int embeddingDimension);

  void InitFakeDB();

  void BulkLoad(ConstArray<KEY_T> keys_array, const void *value);

  void Write(void *h_input, uint64_t startBlock, int block_num);

  // d_d_outputBuffer's size is <ctrl.info.page_size * totalPages>
  double Query(float *d_outputBuffer, const KEY_T *d_index, const int count,
               cudaStream_t stream);

  void PrintResetStats();

  virtual ~BaMWrapper() {}

 private:
  std::vector<Controller *> ctrls_;
  array_t<float> *ssd_array_ = nullptr;
  const uint64_t kDBCapacity;
  const int kEmbeddingDimension;
};

}  // namespace gpu_direct_ssd