#include "kg_controller.h"

using namespace recstore;

namespace recstore {

class CircleBuffer {
 private:
  int L_;
  int start_ = 0;
  int end_ = 0;
  std::vector<c10::intrusive_ptr<SlicedTensor>> sliced_id_tensor_;
  torch::Tensor circle_buffer_end_;
  torch::Tensor circle_buffer_old_end_;
  torch::Tensor step_tensor_;

  std::string backmode_;

 public:
  CircleBuffer(int L, int rank, std::string backmode)
      : L_(L), backmode_(backmode) {
    for (int i = 0; i < L_; i++) {
      sliced_id_tensor_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          folly::sformat("cached_sampler_r{}_{}", rank, i), {int(1e5)},
          torch::kInt64));
    }

    circle_buffer_end_ = IPCTensorFactory::NewIPCTensor(
                             folly::sformat("circle_buffer_end_r{}", rank),
                             {int(1)}, torch::kInt64)
                             .value();

    circle_buffer_old_end_ =
        IPCTensorFactory::NewIPCTensor(
            folly::sformat("circle_buffer_end_cppseen_r{}", rank), {int(1)},
            torch::kInt64)
            .value();

    step_tensor_ =
        IPCTensorFactory::NewIPCTensor(folly::sformat("step_r{}", rank),
                                       {int(L_)}, torch::kInt64)
            .value();

    circle_buffer_end_.fill_(0);
    circle_buffer_old_end_.fill_(0);
  }

  void Push(int step, torch::Tensor item) {
    sliced_id_tensor_[end_]->Copy_(item, false);
    step_tensor_[end_] = step;

    end_ = (end_ + 1) % L_;
    circle_buffer_end_[0] = end_;

    if (backmode_ == "CppAsync") {
      while (circle_buffer_end_[0].item<int64_t>() !=
             circle_buffer_old_end_[0].item<int64_t>()) {
        FB_LOG_EVERY_MS(INFO, 1000) << folly::sformat(
            "Waiting for CppAsync to finish processing the item {}",
            circle_buffer_old_end_[0].item<int64_t>());
      }
    }

    if (end_ == start_) start_ = (start_ + 1) % L_;
  }

  c10::intrusive_ptr<SlicedTensor> Pop() {
    if (start_ == end_) {
      return nullptr;
    }
    auto item = sliced_id_tensor_[start_];
    start_ = (start_ + 1) % L_;
    return item;
  }
};

class BasePerfSampler {
 public:
  BasePerfSampler(int rank, int L, int num_ids_per_step,
                  int64_t full_emb_capacity, std::string backmode)
      : rank_(rank),
        L_(L),
        ids_circle_buffer_(L, rank, backmode),
        sampler_iter_num_(0),
        num_ids_per_step_(num_ids_per_step),
        full_emb_capacity_(full_emb_capacity),
        backmode_(backmode) {}

  void Prefill() {
    for (int i = 0; i < L_; ++i) {
      torch::Tensor entity_id = gen_next_sample();
      ids_circle_buffer_.Push(sampler_iter_num_, entity_id);
      ++sampler_iter_num_;
    }
  }

  torch::Tensor __next__() {
    auto entity_id = gen_next_sample();
    ids_circle_buffer_.Push(sampler_iter_num_, entity_id);
    ++sampler_iter_num_;

    auto ret = ids_circle_buffer_.Pop();

    return ret->GetSlicedTensor();
  }

 protected:
  virtual torch::Tensor gen_next_sample() = 0;

  int rank_;
  int L_;
  CircleBuffer ids_circle_buffer_;
  int sampler_iter_num_;
  int num_ids_per_step_;
  int64_t full_emb_capacity_;
  std::string backmode_;
};

class TestPerfSampler : public BasePerfSampler {
 public:
  TestPerfSampler(int rank, int L, int num_ids_per_step, int full_emb_capacity,
                  std::string backmode)
      : BasePerfSampler(rank, L, num_ids_per_step, full_emb_capacity,
                        backmode) {}

 protected:
  torch::Tensor gen_next_sample() override {
    return torch::randint(0, full_emb_capacity_, {num_ids_per_step_}).cuda();
  }
};

class VirtualEnvironment {
 private:
  int num_gpus_;
  int emb_dim_;
  int64_t cached_capacity_;
  int64_t full_emb_capacity_;
  int L_;
  std::string backmode_;
  int num_ids_per_step_;

  std::vector<torch::Tensor> embedding_cache_;
  torch::Tensor full_emb_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> input_keys_neg_;

  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_;
  std::vector<c10::intrusive_ptr<SlicedTensor>> backward_grads_neg_;

  std::vector<TestPerfSampler> test_perf_sampler_;

  std::vector<std::thread> threads_;

  base::Barrier *barrier_;

 public:
  VirtualEnvironment(const std::string &json_str) {
    auto json_config = json::parse(json_str);
    num_gpus_ = json_config.at("num_gpus");
    emb_dim_ = json_config.at("emb_dim");
    cached_capacity_ = 10;
    full_emb_capacity_ = json_config.at("full_emb_capacity");
    ;
    L_ = json_config.at("L");
    backmode_ = json_config.at("BackwardMode");

    barrier_ = new base::Barrier(num_gpus_);

    CreateVirutalEnvironment();
  }

  void CreateVirutalEnvironment() {
    full_emb_ = IPCTensorFactory::NewIPCTensor(
                    "full_emb", {full_emb_capacity_, emb_dim_}, torch::kFloat32)
                    .value();

    for (int rank = 0; rank < num_gpus_; rank++) {
      embedding_cache_.push_back(IPCTensorFactory::NewIPCGPUTensor(
          folly::sformat("embedding_cache_{}", rank),
          {cached_capacity_, emb_dim_}, torch::kFloat32, rank));

      input_keys_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          folly::sformat("input_keys_{}", rank), {int(1e5)}, torch::kInt64));
      input_keys_neg_.push_back(IPCTensorFactory::NewSlicedIPCTensor(
          folly::sformat("input_keys_neg_{}", rank), {int(1e5)},
          torch::kInt64));

      backward_grads_.push_back(IPCTensorFactory::NewSlicedIPCGPUTensor(
          folly::sformat("backward_grads_{}", rank), {int(1e5), emb_dim_},
          torch::kFloat32, rank));
      backward_grads_neg_.push_back(IPCTensorFactory::NewSlicedIPCGPUTensor(
          folly::sformat("backward_grads_neg_{}", rank), {int(1e5), emb_dim_},
          torch::kFloat32, rank));

      test_perf_sampler_.emplace_back(rank, L_, num_ids_per_step_,
                                      full_emb_capacity_, backmode_);
    }
  }

  void PrefillSampler() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      test_perf_sampler_[rank].Prefill();
    }
  }

  void StartThreads() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      threads_.emplace_back(&VirtualEnvironment::RunThread, this, rank);
    }
  }

  void StopThreads() {
    for (int rank = 0; rank < num_gpus_; rank++) {
      threads_[rank].join();
    }
  }

 private:
  void RunThread(int rank) {
    KGCacheController *controller = KGCacheController::GetInstance();
    cudaSetDevice(rank);
    int step_no = 0;
    while (true) {
      // 1. Get the next step
      auto next_ids = test_perf_sampler_[rank].__next__();
      LOG(INFO) << "Step " << step_no;

      // 2. Forward

      // 3. Backward

      // 4. Update

      barrier_->Wait();
      if (rank == 0) {
        LOG(INFO) << "rank 0 ProcessOneStep";
        controller->ProcessOneStep(step_no);
      }
      barrier_->Wait();
      step_no++;
      if (rank == 0) controller->BlockToStepN(step_no);
      barrier_->Wait();
    }
  }
};

}  // namespace recstore

int main(int argc, char **argv) {
  folly::init(&argc, &argv);
  std::string json_str = R"({
            "num_gpus": 2,
            "L": 10,
            "kForwardItersPerStep": 1,
            "clr": 2,
            "BackwardMode": "CppAsync",
            "nr_background_threads": 32,
            "full_emb_capacity": 1000000,
            "emb_dim" : 2
        })";

  IPCTensorFactory::ClearIPCMemory();

  VirtualEnvironment env(json_str);
  KGCacheController::Init(json_str, {{0, 10}, {10, 20}}, 1000000);
  KGCacheController *controller = KGCacheController::GetInstance();
  controller->RegTensorsPerProcess();
  env.PrefillSampler();
  env.StartThreads();
  env.StopThreads();
  return 0;
}