#pragma once

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
#include "grad_base.h"
#include "parallel_pq.h"
#include "torch_utils.h"

namespace recstore {
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
  static constexpr bool kUseBackThread = false;

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
    xmh::Timer timer_ShuffleKeysAndGrads("ProcessBack:Shuffle");
    auto [shuffled_keys_in_each_rank_cache, shuffled_grads_in_each_rank_cache] =
        ShuffleKeysAndGrads(input_keys, input_grads);
    timer_ShuffleKeysAndGrads.end();

    xmh::Timer timer_SyncUpdateCache("ProcessBack:UpdateCache");
    SyncUpdateCache(shuffled_keys_in_each_rank_cache,
                    shuffled_grads_in_each_rank_cache);
    timer_SyncUpdateCache.end();

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
    xmh::Timer timer_ProcessBackwardAsync("ProcessBack:UpsertPq");
    // #pragma omp parallel for num_threads(num_gpus_)
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

}  // namespace recstore