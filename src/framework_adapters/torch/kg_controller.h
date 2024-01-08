#pragma once
#include <ATen/cuda/CUDAContext.h>
#include <folly/Format.h>
#include <folly/ProducerConsumerQueue.h>
#include <folly/system/MemoryMapping.h>
// #include <oneapi/tbb/concurrent_priority_queue.h>
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
#include "parallel_pq.h"
#include "torch_utils.h"

// #define XMH_DEBUG_KG

namespace recstore {

static constexpr bool kUseBackThread = false;

void RegisterKGCacheController(torch::Library &m);

class GraphEnv {
 public:
  static GraphEnv *instance_;

  static void Init(const std::string &json_str,
                   const std::vector<std::vector<int64_t>> &cached_range,
                   int64_t nr_graph_node) {
    if (instance_ == nullptr)
      instance_ = new GraphEnv(json_str, cached_range, nr_graph_node);
  }

  static GraphEnv *GetInstance() {
    CHECK(instance_ != nullptr);
    return instance_;
  }

 private:
  GraphEnv(const std::string &json_str,
           const std::vector<std::vector<int64_t>> &cached_range,
           int64_t nr_graph_node) {
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config.at("num_gpus");
    L_ = json_config.at("L");
    kForwardItersPerStep_ = json_config.at("kForwardItersPerStep");
    clr_ = json_config.at("clr");

    full_emb_ = IPCTensorFactory::FindIPCTensorFromName("full_emb");

    nr_graph_node_ = nr_graph_node;
    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);
  }

 public:
  void RegTensorsPerProcess() {
    LOG(INFO) << "GraphEnv RegTensorsPerProcess";
    // IPCTensorFactory::ListIPCTensors();
    for (int rank = 0; rank < num_gpus_; ++rank) {
      nv::CudaDeviceRestorer _;
      CUDA_CHECK(cudaSetDevice(rank));
      input_keys_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_{}", rank)));
      input_keys_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("input_keys_neg_{}", rank)));

      // cache tensor
      cache_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("embedding_cache_{}", rank)));

      // L buffer of input ids
      cached_id_circle_buffer_.push_back(
          std::vector<c10::intrusive_ptr<SlicedTensor>>());
      for (int j = 0; j < L_; ++j) {
        cached_id_circle_buffer_[rank].push_back(
            IPCTensorFactory::GetSlicedIPCTensorFromName(
                folly::sformat("cached_sampler_r{}_{}", rank, j)));
      }
      // step tensor
      step_tensor_per_rank_.push_back(IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("step_r{}", rank)));

      backward_grads_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_{}", rank)));

      backward_grads_neg_per_rank_.push_back(
          IPCTensorFactory::GetSlicedIPCTensorFromName(
              folly::sformat("backward_grads_neg_{}", rank)));

      circle_buffer_end_per_rank_.push_back(
          IPCTensorFactory::FindIPCTensorFromName(
              folly::sformat("circle_buffer_end_r{}", rank)));
    }
  }

 public:
  //  config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;
  int64_t nr_graph_node_;

  // state tensor
  torch::Tensor full_emb_;
  std::vector<torch::Tensor> cache_per_rank_;
  // runtime tensor
  std::vector<torch::Tensor> step_tensor_per_rank_;
  std::vector<torch::Tensor> circle_buffer_end_per_rank_;

  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_per_rank_;
  // different rank's L buffer
  std::vector<std::vector<c10::intrusive_ptr<SlicedTensor>>>
      cached_id_circle_buffer_;
};

class GradProcessingBase {
 public:
  GradProcessingBase(const std::string &json_str,
                     const std::vector<std::vector<int64_t>> &cached_range)
      : full_emb_(GraphEnv::GetInstance()->full_emb_),
        cache_per_rank_(GraphEnv::GetInstance()->cache_per_rank_),
        step_tensor_per_rank_(GraphEnv::GetInstance()->step_tensor_per_rank_),
        circle_buffer_end_per_rank_(
            GraphEnv::GetInstance()->circle_buffer_end_per_rank_),
        input_keys_per_rank_(GraphEnv::GetInstance()->input_keys_per_rank_),
        input_keys_neg_per_rank_(
            GraphEnv::GetInstance()->input_keys_neg_per_rank_),
        backward_grads_per_rank_(
            GraphEnv::GetInstance()->backward_grads_per_rank_),
        backward_grads_neg_per_rank_(
            GraphEnv::GetInstance()->backward_grads_neg_per_rank_),
        cached_id_circle_buffer_(
            GraphEnv::GetInstance()->cached_id_circle_buffer_) {
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config.at("num_gpus");
    L_ = json_config.at("L");
    kForwardItersPerStep_ = json_config.at("kForwardItersPerStep");
    clr_ = json_config.at("clr");
    LOG(WARNING) << "Init GradProcessingBase done";
  }

  virtual void RegTensorsPerProcess() {
    GraphEnv::GetInstance()->RegTensorsPerProcess();
    CHECK(!isInitialized_);
    isInitialized_ = true;
  }

  static std::vector<torch::Tensor> split_keys_to_shards(
      const torch::Tensor keys,
      std::vector<std::vector<int64_t>> cached_range) {
    std::vector<torch::Tensor> in_each_rank_cache_mask;
    int num_gpus = cached_range.size();
    for (int shard_no = 0; shard_no < num_gpus; shard_no++) {
      int64_t start = cached_range[shard_no][0];
      int64_t end = cached_range[shard_no][1];

      torch::Tensor in_this_rank =
          keys.greater_equal(start).logical_and(keys.less(end));
      in_each_rank_cache_mask.push_back(in_this_rank);
    }
    return in_each_rank_cache_mask;
  }

  virtual void StopThreads(){

  };

  virtual void ProcessOneStep(int64_t step_no) = 0;

  virtual void BlockToStepN(int step_no) = 0;

  //  cached_range:
  //        [ (start, end ),  # rank0
  //         (start, end),    # rank1
  //         (start, end),  ....
  //         (start, end),
  //         (start, end),    # rank7
  //        ]
  std::pair<std::vector<std::vector<torch::Tensor>>,
            std::vector<std::vector<torch::Tensor>>>
  ShuffleKeysAndGrads(const std::vector<torch::Tensor> &input_keys,
                      const std::vector<torch::Tensor> &input_grads) {
    std::vector<std::vector<torch::Tensor>> shuffled_keys_in_each_rank_cache;
    std::vector<std::vector<torch::Tensor>> shuffled_grads_in_each_rank_cache;
    shuffled_keys_in_each_rank_cache.resize(num_gpus_);
    shuffled_grads_in_each_rank_cache.resize(num_gpus_);

    for (int rank = 0; rank < num_gpus_; rank++) {
      // CHECK(!input_keys[rank].is_cuda());
      auto in_each_rank_cache_mask =
          split_keys_to_shards(input_keys[rank], cached_range_);

      std::vector<torch::Tensor> sharded_keys_in_this_rank =
          TensorUtil::IndexVectors(input_keys[rank], in_each_rank_cache_mask);
      std::vector<torch::Tensor> sharded_grads_in_this_rank =
          TensorUtil::IndexVectors(input_grads[rank], in_each_rank_cache_mask);

      // shuffle keys and grads
      for (int i = 0; i < num_gpus_; i++) {
        shuffled_keys_in_each_rank_cache[i].push_back(
            sharded_keys_in_this_rank[i]);
        shuffled_grads_in_each_rank_cache[i].push_back(
            sharded_grads_in_this_rank[i]);
      }
    }
    // shuffle done
    return std::make_pair(shuffled_keys_in_each_rank_cache,
                          shuffled_grads_in_each_rank_cache);
  }

  void SyncUpdateCache(const std::vector<std::vector<torch::Tensor>>
                           &shuffled_keys_in_each_rank_cache,
                       const std::vector<std::vector<torch::Tensor>>
                           &shuffled_grads_in_each_rank_cache) {
    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // update per rank cache
        // CUDA_CHECK(cudaSetDevice(rank));
        torch::Device device(torch::kCUDA, rank);
        auto in_rank_keys =
            shuffled_keys_in_each_rank_cache[rank][j].to(device, true) -
            cached_range_[rank][0];
        auto shuffle_grads_cuda =
            shuffled_grads_in_each_rank_cache[rank][j].to(device, true);
        cache_per_rank_[rank].index_add_(0, in_rank_keys,
                                         -clr_ * shuffle_grads_cuda);
      }
    }
  }

 protected:
  // config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;

  // state tensor
  torch::Tensor &full_emb_;
  std::vector<torch::Tensor> &cache_per_rank_;
  bool isInitialized_ = false;

  // runtime tensor
  std::vector<torch::Tensor> &step_tensor_per_rank_;
  std::vector<torch::Tensor> &circle_buffer_end_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &input_keys_neg_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> &backward_grads_neg_per_rank_;
  // different rank's L buffer
  std::vector<std::vector<c10::intrusive_ptr<SlicedTensor>>>
      &cached_id_circle_buffer_;
};

class GradSyncProcessing : public GradProcessingBase {
 public:
  GradSyncProcessing(const std::string &json_str,
                     const std::vector<std::vector<int64_t>> &cached_range)
      : GradProcessingBase(json_str, cached_range) {}

  void ProcessOneStep(int64_t step_no) override {
    torch::AutoGradMode guard_false(false);
    // show tensors of the first rank

    // std::cout << "input_keys_per_rank" << std::endl;
    // std::cout << toString(input_keys_per_rank_[0]) << std::endl;
    // std::cout << "input_keys_neg_per_rank_" << std::endl;
    // std::cout << toString(input_keys_neg_per_rank_[0]) << std::endl;
    // static int cnt = 0;
    // std::cout << "cached_id_circle_buffer" << std::endl;
    // std::cout << "Step " << step_tensor_per_rank_[0][cnt].item<int64_t>()
    //           << " ";
    // std::cout <<
    // toString(cached_id_circle_buffer_[0][cnt]->GetSlicedTensor())
    //           << std::endl;
    // cnt = (cnt + 1) % cached_id_circle_buffer_[0].size();

    auto input_keys_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(input_keys_per_rank_);
    auto backward_grads_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(backward_grads_per_rank_);

    ProcessBackwardSync(input_keys_per_rank_tensors,
                        backward_grads_per_rank_tensors);
    if (kForwardItersPerStep_ > 1) {
      auto input_keys_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(input_keys_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_per_rank_);
      ProcessBackwardSync(input_keys_neg_per_rank_tensors,
                          backward_grads_neg_per_rank_tensors);
    }
  }

  void ProcessBackwardSync(const std::vector<torch::Tensor> &input_keys,
                           const std::vector<torch::Tensor> &input_grads) {
    auto [shuffled_keys_in_each_rank_cache, shuffled_grads_in_each_rank_cache] =
        ShuffleKeysAndGrads(input_keys, input_grads);
    SGDGradUpdate(shuffled_keys_in_each_rank_cache,
                  shuffled_grads_in_each_rank_cache, input_keys, input_grads);
  }

  void BlockToStepN(int step_no) {}

  void SGDGradUpdate(const std::vector<std::vector<torch::Tensor>>
                         &shuffled_keys_in_each_rank_cache,
                     const std::vector<std::vector<torch::Tensor>>
                         &shuffled_grads_in_each_rank_cache,
                     const std::vector<torch::Tensor> &input_keys,
                     const std::vector<torch::Tensor> &input_grads) {
    SyncUpdateCache(shuffled_keys_in_each_rank_cache,
                    shuffled_grads_in_each_rank_cache);
#if 0
    for (int rank = 0; rank < num_gpus_; rank++) {
      for (int j = 0; j < num_gpus_; j++) {
        // LOG(WARNING) << rank << ": shuffled_keys_in_each_rank_cache[rank][j]
        // shape="
        //              << shuffled_keys_in_each_rank_cache[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_keys_in_each_rank_cache[rank][j]"
                     << toString(shuffled_keys_in_each_rank_cache[rank][j],
                                 false);
        // LOG(WARNING) << rank << ": shuffled_grads_in_each_rank_cache[rank][j]
        // shape="
        //              << shuffled_grads_in_each_rank_cache[rank][j].sizes();
        LOG(WARNING) << rank << ": shuffled_grads_in_each_rank_cache[rank][j]"
                     << toString(shuffled_grads_in_each_rank_cache[rank][j],
                                 false);

      }
    }
#endif

    // update full emb
    for (int rank = 0; rank < num_gpus_; rank++) {
      full_emb_.index_add_(0, input_keys[rank].cpu(),
                           (-clr_) * input_grads[rank].cpu());
    }
  }
};

class AsyncGradElement {
  static constexpr int kInf = std::numeric_limits<int>::max();

 public:
  AsyncGradElement(int64_t id) : id_(id) { RecaculatePriority(); }

  void MarkReadInStepN(int stepN) {
    base::LockGuard _(lock_);
    read_step_.push_back(stepN);
  }

  void MarkWriteInStepN(int stepN, torch::Tensor grad) {
    base::LockGuard _(lock_);
    write_step_.push_back(stepN);
    write_grad_.push_back(grad);
  }

  int64_t Priority() const {
    CHECK_EQ(magic_, 0xdeadbeef);
    return priority_;
  }

  void RecaculatePriority() {
    base::LockGuard _(lock_);
    int old_priority = priority_;
    if (read_step_.size() == 0) {
      priority_ = kInf;
      CHECK((old_priority <= priority_) || (old_priority == kInf));
      return;
    }
    int min_read_step = MinReadStep();

    if (write_step_.size() == 0) {
      priority_ = kInf;
    } else {
      priority_ = min_read_step;
    }
    CHECK((old_priority <= priority_) || (old_priority == kInf));
  }

  int MinReadStep() const {
    int min_step = *std::min_element(read_step_.begin(), read_step_.end());
    // CHECK_EQ(min_step, read_step_[0]);
    return min_step;
  }

  std::string ToString() const {
    base::LockGuard _(lock_);
    std::stringstream ss;
    ss << folly::sformat("id={}, read_step=[", id_);
    for (auto each : read_step_) {
      ss << each << ",";
    }
    ss << "], write_step=[";
    for (auto each : write_step_) {
      ss << each << ",";
    }
    ss << "], write_grad=[";
    for (auto each : write_grad_) {
      ss << toString(each, false) << ",";
    }
    ss << "], priority=" << Priority();
    return ss.str();
  }

  friend struct CompareAsyncGradElement;

  int64_t GetID() const { return id_; }

  // NOTE: dont use get grad to pass vector<grad> to the controller
  // std::vector<torch::Tensor> GetGrad() const { return write_grad_; }
  std::pair<std::vector<int>, std::vector<torch::Tensor>> DrainWrites() {
    base::LockGuard _(lock_);
    auto ret_write_step = std::move(write_step_);
    auto ret_write_grad = std::move(write_grad_);
    write_step_.clear();
    write_grad_.clear();
    return std::make_pair(ret_write_step, ret_write_grad);
  }

  void RemoveReadStep(int step_no) {
    base::LockGuard _(lock_);
    auto newEnd =
        std::remove_if(read_step_.begin(), read_step_.end(),
                       [step_no](int value) { return value == step_no; });
    read_step_.erase(newEnd, read_step_.end());
  }

  void Lock() { return lock_.Lock(); }

  void Unlock() { return lock_.Unlock(); }

 private:
  int64_t id_;
  std::vector<int> read_step_;
  std::vector<int> write_step_;
  std::vector<torch::Tensor> write_grad_;
  int64_t priority_;
  mutable base::SpinLock lock_;

 public:
  const int magic_ = 0xdeadbeef;
};

struct CompareAsyncGradElement {
  bool operator()(const AsyncGradElement *a, const AsyncGradElement *b) const {
    return a->Priority() > b->Priority();
  }
};

class GradAsyncProcessing : public GradProcessingBase {
  typedef std::pair<int64_t, torch::Tensor> GradWorkTask;
  static constexpr int kInf = std::numeric_limits<int>::max();

 public:
  GradAsyncProcessing(const std::string &json_str,
                      const std::vector<std::vector<int64_t>> &cached_range)
      : GradProcessingBase(json_str, cached_range),
        kEmbNumber_(full_emb_.size(0)),
        kGradDim_(full_emb_.size(1)),
        pq_(kEmbNumber_) {
    auto json_config = json::parse(json_str);
    nr_background_threads_ = json_config.at("nr_background_threads");
    CHECK_GT(nr_background_threads_, 0);
    dict_.assign(full_emb_.size(0), nullptr);
    for (int i = 0; i < full_emb_.size(0); i++)
      dict_[i] = new AsyncGradElement(i);

    if (kUseBackThread) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backthread_work_queues_.emplace_back(
            std::make_unique<folly::ProducerConsumerQueue<GradWorkTask>>(100));
      }
    }

    for (int rank = 0; rank < num_gpus_; rank++) {
      auto ret_tensor = IPCTensorFactory::FindIPCTensorFromName(
          folly::sformat("circle_buffer_end_cppseen_r{}", rank));
      circle_buffer_end_cppseen_.push_back(ret_tensor);
    }
  }

  void StartThreads() {
    CHECK(isInitialized_);
    dispatch_thread_stop_flag_ = false;
    grad_thread_stop_flag_ = false;

    if (kUseBackThread) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_.emplace_back(
            std::bind(&GradAsyncProcessing::GradWorkThread, this, i));
      }
    }
    dispatch_thread_ = std::thread(&GradAsyncProcessing::DispatchThread, this);
  }

  void StopThreads() override {
    CHECK(isInitialized_);

    bool expected = false;
    if (!grad_thread_stop_flag_.compare_exchange_strong(expected, true)) {
      return;
    }

    grad_thread_stop_flag_ = true;
    dispatch_thread_stop_flag_ = true;

    dispatch_thread_.join();
    if (kUseBackThread) {
      for (int i = 0; i < nr_background_threads_; ++i) {
        backward_threads_[i].join();
      }
    }
    LOG(WARNING) << "StopThreads done.";
  }

  void RegTensorsPerProcess() override {
    GradProcessingBase::RegTensorsPerProcess();
    StartThreads();
  }

  void ProcessOneStep(int64_t step_no) override {
    torch::AutoGradMode guard_false(false);
    // show tensors of the first rank

    // std::cout << "input_keys_per_rank" << std::endl;
    // std::cout << toString(input_keys_per_rank_[0]) << std::endl;
    // std::cout << "input_keys_neg_per_rank_" << std::endl;
    // std::cout << toString(input_keys_neg_per_rank_[0]) << std::endl;

    auto input_keys_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(input_keys_per_rank_);
    auto backward_grads_per_rank_tensors =
        SlicedTensor::BatchConvertToTensors(backward_grads_per_rank_);

    ProcessBackwardAsync(input_keys_per_rank_tensors,
                         backward_grads_per_rank_tensors, step_no);
    if (kForwardItersPerStep_ > 1) {
      auto input_keys_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(input_keys_neg_per_rank_);
      auto backward_grads_neg_per_rank_tensors =
          SlicedTensor::BatchConvertToTensors(backward_grads_neg_per_rank_);
      ProcessBackwardAsync(input_keys_neg_per_rank_tensors,
                           backward_grads_neg_per_rank_tensors, step_no);
    }
  }

  void WhenNewSampleComes(c10::intrusive_ptr<SlicedTensor> input_keys, int rank,
                          int step_no) {
    // 来了一个新样本step号：把<里面的ID, step号>插堆
    auto *data = input_keys->GetSlicedTensor().data_ptr<int64_t>();
    for (int i = 0; i < input_keys->GetSlicedTensor().size(0); ++i) {
      int64_t id = data[i];
      auto *p = dict_[id];
      p->MarkReadInStepN(step_no);
      p->RecaculatePriority();
      pq_.PushOrUpdate(p);
    }
  }

  void DetectNewSamplesCome() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      int64_t new_end = circle_buffer_end_per_rank_[rank].item<int64_t>();
      int64_t *p_old_end = circle_buffer_end_cppseen_[rank].data_ptr<int64_t>();
      int64_t old_end = circle_buffer_end_cppseen_[rank][0].item<int64_t>();
      CHECK_EQ(*p_old_end, old_end);
      if (new_end != old_end) {
        FB_LOG_EVERY_MS(WARNING, 1000) << folly::sformat(
            "Detect new sample comes, old_end{}, new_end{}", old_end, new_end);

        // add [circle_buffer_old_end, new_end)
        if (new_end < old_end) new_end += L_;
        for (int i = old_end; i < new_end; ++i) {
          int pointer = (i % L_);
          int step = step_tensor_per_rank_[rank][pointer].item<int64_t>();
          WhenNewSampleComes(cached_id_circle_buffer_[rank][pointer], rank,
                             step);
        }
        new_end = new_end % L_;
        *p_old_end = new_end;
      }
    }
  }

  void DispatchThread() {
    CHECK(isInitialized_);
    while (!dispatch_thread_stop_flag_.load()) {
      base::LockGuard _(large_lock_);
      DetectNewSamplesCome();
      // 后台线程，不断取堆头，dispatch给worker
      if (pq_.empty()) {
        continue;
      }
      auto *p = pq_.top();

      // re-read
      if (!p) continue;
      CHECK_EQ(p->magic_, 0xdeadbeef);
      int64_t id = p->GetID();

      // NOTE: 改了优先级
      auto [not_used, grads] = p->DrainWrites();
      static int round_robin = 0;
      for (int i = 0; i < grads.size(); i++) {
        // constexpr bool kUseBackThread = true;
        if (kUseBackThread) {
          while (!backthread_work_queues_[round_robin]->write(
              std::make_pair(id, grads[i]))) {
            continue;
          }
        } else {
          auto grad = grads[i].cpu().unsqueeze_(0);
#ifdef XMH_DEBUG_KG
          LOG(INFO) << "+Grad: "
                    << "| " << p->ToString() << "|" << toString(full_emb_[id])
                    << " -> "
                    << toString(full_emb_[id] - clr_ * grad.squeeze(0));
#endif
          full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
        }
      }
      // TODO:
      // 其实这里并不能把他删掉，因为如果用后台线程GradWorkThread，后台线程还没做完
      // pq_.pop_x(p);
      // p->RecaculatePriority();

      p->RecaculatePriority();
      pq_.PushOrUpdate(p);

      round_robin = (round_robin + 1) % nr_background_threads_;
    }
  }

  void GradWorkThread(int thread_id) {
    auto *queue = backthread_work_queues_[thread_id].get();
    CHECK(queue != nullptr);
    while (!grad_thread_stop_flag_.load()) {
      std::pair<int64_t, torch::Tensor> p;
      while (!queue->read(p)) {
        // spin until we get a value
        if (grad_thread_stop_flag_.load()) return;
        continue;
      }
      int64_t id = p.first;
      torch::Tensor grad = p.second;
      grad = grad.cpu().unsqueeze_(0);
      full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
    }
  }

  // 等到可以让step_no开始训练
  void BlockToStepN(int step_no) override {
    // 等待堆头的元素大于step_no号
    while (true) {
      base::LockGuard _(large_lock_);

      if (pq_.empty()) {
        LOG(WARNING) << "pq is empty";
        break;
      }

      int priority = pq_.MinPriority();
      if (priority > step_no) {
#ifdef XMH_DEBUG_KG
        LOG(INFO) << folly::sformat("top(pq)'s priority={} > step_no{}.",
                                    priority, step_no)
                  << pq_.ToString();
#endif
        break;
      }
      FB_LOG_EVERY_MS(WARNING, 1000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", pq.top=" << priority;
    }
  }

  void ProcessBackwardAsync(const std::vector<torch::Tensor> &input_keys,
                            const std::vector<torch::Tensor> &input_grads,
                            int step_no) {
    auto [shuffled_keys_in_each_rank_cache, shuffled_grads_in_each_rank_cache] =
        ShuffleKeysAndGrads(input_keys, input_grads);
    SyncUpdateCache(shuffled_keys_in_each_rank_cache,
                    shuffled_grads_in_each_rank_cache);

    // LOG(WARNING) << "shuffled_keys_in_each_rank_cache";
    // LOG(WARNING) << "Rank0:" <<
    // toString(shuffled_keys_in_each_rank_cache[0][0])
    //              << "|" << toString(shuffled_keys_in_each_rank_cache[0][1]);
    // LOG(WARNING) << "Rank1:" <<
    // toString(shuffled_keys_in_each_rank_cache[1][0])
    //              << "|" << toString(shuffled_keys_in_each_rank_cache[1][1]);

    // LOG(WARNING) << "shuffled_grads_in_each_rank_cache";

    // LOG(WARNING) << "Rank0:"
    //              << toString(shuffled_grads_in_each_rank_cache[0][0]) << "|"
    //              << toString(shuffled_grads_in_each_rank_cache[0][1]);
    // LOG(WARNING) << "Rank1:"
    //              << toString(shuffled_grads_in_each_rank_cache[1][0]) << "|"
    //              << toString(shuffled_grads_in_each_rank_cache[1][1]);

    base::LockGuard _(large_lock_);

    // record the update
    // 把 <ID>查一下堆，拿一下step号
    // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
    // 如果在堆，建立映射，把grad指针填进去
    xmh::Timer timer_ProcessBackwardAsync("ProcessBackwardAsync");

#pragma omp parallel for num_threads(num_gpus_)
    for (int rank = 0; rank < input_keys.size(); ++rank) {
      auto *data = input_keys[rank].data_ptr<int64_t>();
      CHECK(input_keys[rank].is_cpu());
      CHECK_EQ(input_grads[rank].dim(), 2);

      for (int i = 0; i < input_keys[rank].size(0); ++i) {
        int64_t id = data[i];
        torch::Tensor grad_tensor = input_grads[rank][i];
        auto *p = dict_[id];
        // NOTE: 改了优先级
        p->RemoveReadStep(step_no);
        p->MarkWriteInStepN(step_no, grad_tensor);
        p->RecaculatePriority();
#ifdef XMH_DEBUG_KG
        LOG(INFO) << folly::sformat("Push pq_ | id={}, step_no={}, grad={}", id,
                                    step_no, toString(grad_tensor, false));
#endif
        pq_.PushOrUpdate(p);
      }
    }
    timer_ProcessBackwardAsync.end();

    // LOG(WARNING) << "<ProcessBackwardAsync>" << pq_.ToString();
  }

  void PrintPq() {
    base::LockGuard _(large_lock_);
    //   LOG(ERROR) << pq_.ToString();
  }

 private:
  int nr_background_threads_;
  std::vector<AsyncGradElement *> dict_;

  const int64_t kEmbNumber_;
  const int kGradDim_;

  // base::CustomPriorityQueue<AsyncGradElement *, CompareAsyncGradElement> pq_;
  recstore::ParallelPq<AsyncGradElement *> pq_;

  std::thread dispatch_thread_;
  std::vector<std::thread> backward_threads_;
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<GradWorkTask>>>
      backthread_work_queues_;

  std::vector<torch::Tensor> circle_buffer_end_cppseen_;

  // base::SpinLock large_lock_;
  base::PlaceboLock large_lock_;

  std::atomic_bool dispatch_thread_stop_flag_{false};
  std::atomic_bool grad_thread_stop_flag_{false};
};

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
    L_ = json_config.at("L");
    kForwardItersPerStep_ = json_config.at("kForwardItersPerStep");
    clr_ = json_config.at("clr");

    auto backward_mode = json_config.at("BackwardMode");

    if (backward_mode == "CppSync") {
      grad_processing_ = new GradSyncProcessing(json_str, cached_range);
      LOG(WARNING) << "after init GradSyncProcessing";
    } else if (backward_mode == "CppAsync") {
      grad_processing_ = new GradAsyncProcessing(json_str, cached_range);
      LOG(WARNING) << "after init GradAsyncProcessing";
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
    grad_processing_->ProcessOneStep(step_no);
  }

  void BlockToStepN(int64_t step_no) {
    grad_processing_->BlockToStepN(step_no);
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