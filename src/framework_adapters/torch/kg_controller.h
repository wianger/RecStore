#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <folly/Format.h>
#include <folly/ProducerConsumerQueue.h>
#include <folly/system/MemoryMapping.h>
// #include <oneapi/tbb/concurrent_priority_queue.h>
#include <gperftools/profiler.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "IPCTensor.h"
#include "base/base.h"
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/json.h"
#include "base/lock.h"
#include "base/pq.h"
#include "base/timer.h"
#include "grad_async_v1.h"
#include "grad_async_v2.h"
#include "grad_base.h"
#include "parallel_pq.h"
#include "torch_utils.h"

// #define XMH_DEBUG_KG

namespace recstore {

void RegisterKGCacheController(torch::Library &m);

class KGCacheController : public torch::CustomClassHolder {
  static KGCacheController *instance_;

 public:
  static c10::intrusive_ptr<KGCacheController> Init(
      const std::string &json_str,
      const std::vector<std::vector<int64_t>> &cached_range,
      const int64_t nr_graph_nodes) {
    GraphEnv::Init(json_str, cached_range, nr_graph_nodes);
    return c10::make_intrusive<KGCacheController>(json_str, cached_range);
  }

  static KGCacheController *GetInstance() {
    CHECK(instance_ != nullptr);
    return instance_;
  }

  KGCacheController(const std::string &json_str,
                    const std::vector<std::vector<int64_t>> &cached_range) {
    CHECK(instance_ == nullptr);
    instance_ = this;
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config.at("num_gpus");

    CHECK_EQ(num_gpus_, cached_range.size())
        << "cached ranges in GPUs not match # of GPUs";

    L_ = json_config.at("L");
    kForwardItersPerStep_ = json_config.at("kForwardItersPerStep");
    clr_ = json_config.at("clr");

    auto backward_mode = json_config.at("backwardMode");

    if (backward_mode == "CppSync") {
      grad_processing_ = new GradSyncProcessing(json_str, cached_range);
      LOG(WARNING) << "after init GradSyncProcessing";
    } else if (backward_mode == "CppAsync") {
      grad_processing_ = new GradAsyncProcessing(json_str, cached_range);
      LOG(WARNING) << "after init GradAsyncProcessing";
    } else if (backward_mode == "CppAsyncV2") {
      grad_processing_ = new GradAsyncProcessingV2(json_str, cached_range);
      LOG(WARNING) << "after init GradAsyncProcessingV2";
    } else if (backward_mode == "PySync") {
      ;
    } else {
      LOG(FATAL) << "invalid backward mode: " << backward_mode;
    }
    LOG(INFO) << "Construct KGCacheController done";
  }

 public:
  void RegTensorsPerProcess() { grad_processing_->RegTensorsPerProcess(); }

  void ProcessOneStep(int64_t step_no) {
    xmh::Timer timer_processonestep("ProcessOneStep");
    grad_processing_->ProcessOneStep(step_no);
    timer_processonestep.end();
  }

  void BlockToStepN(int64_t step_no) {
    xmh::Timer timer_blocktostepn("BlockToStepN");
    grad_processing_->BlockToStepN(step_no);
    timer_blocktostepn.end();
  }

  void StopThreads() { grad_processing_->StopThreads(); }

  void PrintPq() const {
    // ((GradAsyncProcessing *)grad_processing_)->PrintPq();
  }

 private:
  GradProcessingBase *grad_processing_;
  // config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;
};

}  // namespace recstore