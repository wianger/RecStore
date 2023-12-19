#include <ATen/cuda/CUDAContext.h>
#include <folly/Format.h>
#include <folly/ProducerConsumerQueue.h>
#include <folly/system/MemoryMapping.h>
#include <oneapi/tbb/concurrent_priority_queue.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "IPCTensor.h"
#include "base/base.h"
#include "base/cu_utils.cuh"
#include "base/debug_utils.h"
#include "base/json.h"
#include "base/pq.h"
#include "torch_utils.h"

namespace recstore {

class GraphEnv {
 public:
  static GraphEnv *instance_;

  static void Init(const std::string &json_str,
                   const std::vector<std::vector<int64_t>> &cached_range) {
    if (instance_ == nullptr) instance_ = new GraphEnv(json_str, cached_range);
  }

  static GraphEnv *GetInstance() {
    CHECK(instance_ != nullptr);
    return instance_;
  }

 private:
  GraphEnv(const std::string &json_str,
           const std::vector<std::vector<int64_t>> &cached_range) {
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config["num_gpus"];
    L_ = json_config["L"];
    kForwardItersPerStep_ = json_config["kForwardItersPerStep"];
    clr_ = json_config["clr"];

    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);
  }

 public:
  void RegTensorsPerProcess() {
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
    }

    full_emb_ = IPCTensorFactory::FindIPCTensorFromName("full_emb");
  }

 public:
  //  config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;

  // state tensor
  torch::Tensor full_emb_;
  std::vector<torch::Tensor> cache_per_rank_;
  // runtime tensor
  std::vector<torch::Tensor> step_tensor_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_per_rank_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_per_rank_;
  // different rank's L buffer
  std::vector<std::vector<c10::intrusive_ptr<SlicedTensor>>>
      cached_id_circle_buffer_;
};

GraphEnv *GraphEnv::instance_;

class GradProcessingBase {
 public:
  GradProcessingBase(const std::string &json_str,
                     const std::vector<std::vector<int64_t>> &cached_range)
      : full_emb_(GraphEnv::GetInstance()->full_emb_),
        cache_per_rank_(GraphEnv::GetInstance()->cache_per_rank_),
        step_tensor_per_rank_(GraphEnv::GetInstance()->step_tensor_per_rank_),
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
    num_gpus_ = json_config["num_gpus"];
    L_ = json_config["L"];
    kForwardItersPerStep_ = json_config["kForwardItersPerStep"];
    clr_ = json_config["clr"];

    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);
    GraphEnv::Init(json_str, cached_range_);
  }

  void RegTensorsPerProcess() {
    GraphEnv::GetInstance()->RegTensorsPerProcess();

    // config
    full_emb_ = GraphEnv::GetInstance()->full_emb_;
    cache_per_rank_ = GraphEnv::GetInstance()->cache_per_rank_;
    // runtime tensor
    step_tensor_per_rank_ = GraphEnv::GetInstance()->step_tensor_per_rank_;
    input_keys_per_rank_ = GraphEnv::GetInstance()->input_keys_per_rank_;
    input_keys_neg_per_rank_ =
        GraphEnv::GetInstance()->input_keys_neg_per_rank_;
    backward_grads_per_rank_ =
        GraphEnv::GetInstance()->backward_grads_per_rank_;
    backward_grads_neg_per_rank_ =
        GraphEnv::GetInstance()->backward_grads_neg_per_rank_;
    // different rank's L buffer
    cached_id_circle_buffer_ =
        GraphEnv::GetInstance()->cached_id_circle_buffer_;
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

  virtual void ProcessOneStep() = 0;

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
        cache_per_rank_[rank].index_add_(0, in_rank_keys, -shuffle_grads_cuda);
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
  // runtime tensor
  std::vector<torch::Tensor> &step_tensor_per_rank_;
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

  void ProcessOneStep() {
    torch::AutoGradMode guard_false(false);
    // show tensors of the first rank
    std::cout << "input_keys_per_rank" << std::endl;
    std::cout << toString(input_keys_per_rank_[0]) << std::endl;
    std::cout << "input_keys_neg_per_rank_" << std::endl;
    std::cout << toString(input_keys_neg_per_rank_[0]) << std::endl;
    static int cnt = 0;
    std::cout << "cached_id_circle_buffer" << std::endl;
    std::cout << "Step " << step_tensor_per_rank_[0][cnt].item<int64_t>()
              << " ";
    std::cout << toString(cached_id_circle_buffer_[0][cnt]->GetSlicedTensor())
              << std::endl;
    cnt = (cnt + 1) % cached_id_circle_buffer_[0].size();

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
      full_emb_.index_add_(0, input_keys[rank].cpu(), -input_grads[rank].cpu());
    }
  }
};

class CustomElement {
 public:
  CustomElement(int64_t id) : id_(id) {}

  void MarkReadInStepN(int stepN) {
    step_.push_back(stepN);
    grad_.push_back(std::nullopt);
  }
  void MarkWriteInStepN(int stepN, torch::Tensor grad) {
    step_.push_back(stepN);
    grad_.push_back(grad);
  }

  int MinStep() const { return *std::min_element(step_.begin(), step_.end()); }

  friend struct CompareCustomElement;

  int64_t GetID() const { return id_; }

  std::vector<int> GetStep() const { return step_; }

  std::vector<std::optional<torch::Tensor>> GetGrad() const { return grad_; }

 private:
  int64_t id_;
  std::vector<int> step_;
  std::vector<std::optional<torch::Tensor>> grad_;
};

struct CompareCustomElement {
  bool operator()(const CustomElement *a, const CustomElement *b) const {
    return a->step_[0] > b->step_[0];
  }
};

class GradAsyncProcessing : public GradProcessingBase {
  typedef std::pair<int64_t, torch::Tensor> GradWorkTask;
  static constexpr int kInf = std::numeric_limits<int>::max();

 public:
  GradAsyncProcessing(const std::string &json_str,
                      const std::vector<std::vector<int64_t>> &cached_range)
      : GradProcessingBase(json_str, cached_range),
        kGradDim_(full_emb_.size(1)) {
    auto json_config = json::parse(json_str);
    nr_background_threads_ = json_config["nr_background_threads"];

    CHECK_GT(nr_background_threads_, 0);
    dict_.assign(full_emb_.size(0), nullptr);
    backthread_work_queues_.emplace_back(
        std::make_unique<folly::ProducerConsumerQueue<GradWorkTask>>(100));
    for (int i = 0; i < nr_background_threads_; ++i) {
      backward_threads_.emplace_back(
          std::bind(&GradAsyncProcessing::BackwardWorkThread, this, i));
    }
    dispatch_thread_ = std::thread(&GradAsyncProcessing::DispatchThread, this);
  }

  void ProcessOneStep() override {}

  void WhenNewSampleComes(c10::intrusive_ptr<SlicedTensor> input_keys, int rank,
                          int step_no) {
    // 来了一个新样本step号：把<里面的ID, step号>插堆
    auto *data = input_keys->GetSlicedTensor().data_ptr<int64_t>();
    for (int i = 0; i < input_keys->GetSlicedTensor().size(0); ++i) {
      int64_t id = data[i];
      if (dict_[id] == nullptr) {
        // 如果不在， 就直接插一个新的
        auto *p = new CustomElement(id);
        p->MarkReadInStepN(step_no);
        pq_.push(p);
        dict_[id] = p;
      } else {
        // 如果在堆里，那就补一个新的轮次进去，比如[step 3, step 6]用到了,
        // 补成[3, 6, step_no], 对应的grad应该是[XXX, XXX, USE(nullopt)]
        auto *p = dict_[id];
        p->MarkReadInStepN(step_no);
      }
    }
  }

  void DispatchThread() {
    // 后台线程，不断取堆头，dispatch给worker
    while (true) {
      auto *p = pq_.top();
      int64_t id = p->GetID();
      auto steps = p->GetStep();

      int step_no = p->MinStep();
      torch::Tensor grad = full_emb_.index({id});
    }
  }

  void BackwardWorkThread(int thread_id) {
    auto *queue = backthread_work_queues_[thread_id].get();
    while (true) {
      std::pair<int64_t, torch::Tensor> p;
      while (!queue->read(p)) {
        // spin until we get a value
        continue;
      }
      int64_t id = p.first;
      torch::Tensor grad = p.second;
      full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
    }
  }

  void BlockToStepN(int step_no) {
    // 等待堆头的元素大于step_no号
    while (true) {
      auto *p = pq_.top();
      if (p->MinStep() > step_no) break;
      FB_LOG_EVERY_MS(INFO, 10000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", min_step=" << p->MinStep();
    }
  }

  void ProcessBackwardAsync(const std::vector<torch::Tensor> &input_keys,
                            const std::vector<torch::Tensor> &input_grads,
                            int step_no) {
    auto [shuffled_keys_in_each_rank_cache, shuffled_grads_in_each_rank_cache] =
        ShuffleKeysAndGrads(input_keys, input_grads);
    SyncUpdateCache(shuffled_keys_in_each_rank_cache,
                    shuffled_grads_in_each_rank_cache);
    // TODO: record the update

    // 把 <ID>查一下堆，拿一下step号
    // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
    // 如果在堆，建立映射，把grad指针填进去
    for (int rank = 0; rank < input_keys.size(); ++rank) {
      auto *data = input_keys[rank].data_ptr<int64_t>();
      CHECK(!input_keys[rank].is_cuda());
      CHECK_EQ(input_grads[rank].dim(), 2);

      for (int i = 0; i < input_keys[rank].size(0); ++i) {
        int64_t id = data[i];
        torch::Tensor grad_tensor = input_grads[rank][i];
        if (dict_[id] == nullptr) {
          // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
          auto *p = new CustomElement(id);
          p->MarkWriteInStepN(kInf, grad_tensor);
          pq_.push(p);
          dict_[id] = p;
        } else {
          // 如果在堆里，那就补一个新的轮次进去，
          // 比如[step 3, step 6]用到了, 补成[3, 6, step_no]
          // 比如[正无穷], 改成[正无穷, step_no]
          auto *p = dict_[id];
          p->MarkWriteInStepN(step_no, grad_tensor);
          // TODO:此时优先级改了，需要调整最小堆
          pq_.adjustPriority(p);
        }
      }
    }
  }

 private:
  int nr_background_threads_;

  std::vector<CustomElement *> dict_;
  // oneapi::tbb::concurrent_priority_queue<CustomElement *,
  // CompareCustomElement>
  //     pq_;
  base::CustomPriorityQueue<CustomElement *, CompareCustomElement> pq_;
  std::thread dispatch_thread_;
  std::vector<std::thread> backward_threads_;
  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<GradWorkTask>>>
      backthread_work_queues_;
  const int kGradDim_;
};

class KGCacheController : public torch::CustomClassHolder {
  static KGCacheController *instance_;

 public:
  static c10::intrusive_ptr<KGCacheController> Init(
      const std::string &json_str,
      const std::vector<std::vector<int64_t>> &cached_range) {
    GraphEnv::Init(json_str, cached_range);
    return c10::make_intrusive<KGCacheController>(json_str, cached_range);
  }

  static KGCacheController *GetInstance() {
    CHECK(instance_ != nullptr);
    return instance_;
  }

  KGCacheController(const std::string &json_str,
                    const std::vector<std::vector<int64_t>> &cached_range) {
    CHECK(instance_ == nullptr);
    cached_range_ = cached_range;
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config["num_gpus"];
    L_ = json_config["L"];
    kForwardItersPerStep_ = json_config["kForwardItersPerStep"];
    clr_ = json_config["clr"];

    LOG(WARNING) << folly::sformat("KGCacheController, config={}", json_str);

    grad_processing_ = new GradSyncProcessing(json_str, cached_range);
    grad_processing_ = new GradAsyncProcessing(json_str, cached_range);
  }

 public:
  void RegTensorsPerProcess() { grad_processing_->RegTensorsPerProcess(); }

  void ProcessOneStep() { grad_processing_->ProcessOneStep(); }

 private:
  GradProcessingBase *grad_processing_;
  // config
  int num_gpus_;
  int L_;
  std::vector<std::vector<int64_t>> cached_range_;
  int kForwardItersPerStep_;
  float clr_;
};

#if 0
class AsyncFlusher {
  typedef std::pair<int64_t, torch::Tensor> GradWorkTask;

 public:
  AsyncFlusher(int nr_background_threads, float clr)
      : nr_background_threads_(nr_background_threads),
        clr_(clr),
        full_emb_(GraphEnv::GetInstance()->full_emb_),
        kGradDim_(full_emb_.size(1)) {
    CHECK_GT(nr_background_threads_, 0);
    dict_.assign(full_emb_.size(0), nullptr);

    backthread_work_queues_.emplace_back(
        std::make_unique<folly::ProducerConsumerQueue<GradWorkTask>>(100));
    for (int i = 0; i < nr_background_threads_; ++i) {
      backward_threads_.emplace_back(
          std::bind(&AsyncFlusher::BackwardWorkThread, this, i));
    }
    dispatch_thread_ = std::thread(&AsyncFlusher::DispatchThread, this);
  }

  void WhenNewSampleComes(c10::intrusive_ptr<SlicedTensor> input_keys, int rank,
                          int step_no) {
    // 来了一个新样本step号：把<里面的ID, step号>插堆
    auto *data = input_keys->GetSlicedTensor().data_ptr<int64_t>();
    for (int i = 0; i < input_keys->GetSlicedTensor().size(0); ++i) {
      int64_t id = data[i];
      if (dict_[id] == nullptr) {
        // 如果不在， 就直接插一个新的
        auto *p = new CustomElement(id);
        p->MarkReadInStepN(step_no);
        pq_.push(p);
        dict_[id] = p;
      } else {
        // 如果在堆里，那就补一个新的轮次进去，比如[step 3, step 6]用到了,
        // 补成[3, 6, step_no], 对应的grad应该是[XXX, XXX, USE(nullopt)]
        auto *p = dict_[id];
        p->MarkReadInStepN(step_no);
      }
    }
  }

  void DispatchThread() {
    // 后台线程，不断取堆头，dispatch给worker
    while (true) {
      auto *p = pq_.top();
      int64_t id = p->GetID();
      auto steps = p->GetStep();

      int step_no = p->MinStep();
      torch::Tensor grad = full_emb_.index({id});
    }
  }

  void BackwardWorkThread(int thread_id) {
    auto *queue = backthread_work_queues_[thread_id].get();
    while (true) {
      std::pair<int64_t, torch::Tensor> p;
      while (!queue->read(p)) {
        // spin until we get a value
        continue;
      }
      int64_t id = p.first;
      torch::Tensor grad = p.second;
      full_emb_.index_add_(0, torch::full({1}, id), -clr_ * grad);
    }
  }

  void BlockToStepN(int step_no) {
    // 等待堆头的元素大于step_no号
    while (true) {
      auto *p = pq_.top();
      if (p->MinStep() > step_no) break;
      FB_LOG_EVERY_MS(INFO, 10000)
          << "Sleep in <BlockToStepN>, step_no=" << step_no
          << ", min_step=" << p->MinStep();
    }
  }

  void ProcessBackwardAsync(const std::vector<torch::Tensor> &input_keys,
                            const std::vector<torch::Tensor> &input_grads,
                            int step_no) {
    // 把 <ID>查一下堆，拿一下step号
    // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
    // 如果在堆，建立映射，把grad指针填进去
    for (int rank = 0; rank < input_keys.size(); ++rank) {
      auto *data = input_keys[rank].data_ptr<int64_t>();
      CHECK(!input_keys[rank].is_cuda());
      CHECK_EQ(input_grads[rank].dim(), 2);

      for (int i = 0; i < input_keys[rank].size(0); ++i) {
        int64_t id = data[i];
        torch::Tensor grad_tensor = input_grads[rank][i];
        if (dict_[id] == nullptr) {
          // 如果不在堆，就插堆<ID, +无穷>，把grad指针填进去
          auto *p = new CustomElement(id);
          p->MarkWriteInStepN(kInf, grad_tensor);
          pq_.push(p);
          dict_[id] = p;
        } else {
          // 如果在堆里，那就补一个新的轮次进去，
          // 比如[step 3, step 6]用到了, 补成[3, 6, step_no]
          // 比如[正无穷], 改成[正无穷, step_no]
          auto *p = dict_[id];
          p->MarkWriteInStepN(step_no, grad_tensor);
          // TODO:此时优先级改了，需要调整最小堆
          pq_.adjustPriority(p);
        }
      }
    }
  }

 private:
  std::vector<CustomElement *> dict_;

  // oneapi::tbb::concurrent_priority_queue<CustomElement *,
  // CompareCustomElement>
  //     pq_;
  base::CustomPriorityQueue<CustomElement *, CompareCustomElement> pq_;

  std::thread dispatch_thread_;
  std::vector<std::thread> backward_threads_;

  std::vector<std::unique_ptr<folly::ProducerConsumerQueue<GradWorkTask>>>
      backthread_work_queues_;

  torch::Tensor &full_emb_;
  const int kGradDim_;
};
#endif

KGCacheController *KGCacheController::instance_;

void RegisterKGCacheController(torch::Library &m) {
  m.class_<KGCacheController>("KGCacheController")
      .def_static("Init", &KGCacheController::Init)
      // .def(torch::init<const std::string,
      //                  const std::vector<std::vector<int64_t>>>())
      .def("RegTensorsPerProcess", &KGCacheController::RegTensorsPerProcess)
      .def("ProcessOneStep", &KGCacheController::ProcessOneStep);
}

}  // namespace recstore
