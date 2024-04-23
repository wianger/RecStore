#pragma once

#include <folly/system/MemoryMapping.h>
#include <immintrin.h>
#include <torch/custom_class.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "base/lock.h"
#include "base/log.h"
#include "base/math.h"
#include "base/pprint.h"
#include "base/sleep.h"

namespace recstore {
class MultiProcessBarrier {
  explicit MultiProcessBarrier(const std::string &name, int count)
      : count_(count), bar_(0) {
    strcpy(name_, name.c_str());
  }

 public:
  void Wait() {
    int passed_old = passed_.load(std::memory_order_relaxed);

    if (bar_.fetch_add(1) == (count_ - 1)) {
      // The last thread, faced barrier.
      bar_ = 0;
      // Synchronize and store in one operation.
      passed_.store(passed_old + 1, std::memory_order_release);
    } else {
      // Not the last thread. Wait others.
      while (passed_.load(std::memory_order_relaxed) == passed_old) {
      };
      // Need to synchronize cache with other threads, passed barrier.
      std::atomic_thread_fence(std::memory_order_acquire);
    }
  }

  friend class MultiProcessBarrierFactory;

 private:
  char name_[256];
  int count_;
  std::atomic_int bar_;
  std::atomic_int passed_{0};
};

class MultiProcessBarrierHolder : public torch::CustomClassHolder {
 public:
  MultiProcessBarrierHolder(MultiProcessBarrier *barrier) : barrier_(barrier) {}

  void Wait() { barrier_->Wait(); }

 private:
  MultiProcessBarrier *barrier_;
};

class MultiProcessBarrierFactory : public torch::CustomClassHolder {
  static constexpr int kMaxRegBarrier = 20;
  struct IPCBarrierShmRegion {
    IPCBarrierShmRegion() {}
    std::atomic<int64_t> counter_;
    base::SpinLock lock_;
    MultiProcessBarrier handles[0];
  };

  MultiProcessBarrierFactory() {
    folly::MemoryMapping::Options options =
        folly::MemoryMapping::writable().setPrefault(true).setShared(true);
    options.address = (void *)(0x080000000000);
    system("touch /dev/shm/recstore_ipc_barrier");
    mapping_ = new folly::MemoryMapping("/dev/shm/recstore_ipc_barrier", 0,
                                        kShmSize, options);
    header_ = new ((IPCBarrierShmRegion *)mapping_->writableRange().begin())
        IPCBarrierShmRegion();
  }

 public:
  static c10::intrusive_ptr<MultiProcessBarrierHolder> CreateStatic(
      const std::string &name, int64_t count) {
    auto *p = MultiProcessBarrierFactory::GetInstance()->Create(name, count);
    return c10::make_intrusive<MultiProcessBarrierHolder>(p);
  }

  //   static MultiProcessBarrier *CreateStatic(const std::string &name,
  //                                            int64_t count) {
  //     return MultiProcessBarrierFactory::GetInstance()->Create(name, count);
  //   }

  static void ClearIPCMemoryStatic() {
    return MultiProcessBarrierFactory::GetInstance()->ClearIPCMemory();
  }

  static MultiProcessBarrierFactory *GetInstance() {
    static MultiProcessBarrierFactory *instance =
        new MultiProcessBarrierFactory();
    return instance;
  }

 private:
  MultiProcessBarrier *Create(const std::string &name, int count) {
    base::LockGuard<base::SpinLock> lock_guard(header_->lock_);

    for (int i = 0; i < header_->counter_; i++) {
      if (strcmp(header_->handles[i].name_, name.c_str()) == 0 &&
          header_->handles[i].count_ == count) {
        LOG(INFO) << "CreateBarrier: find existing barrier, name: " << name
                  << ", count: " << count;
        return &header_->handles[i];
      }
    }

    LOG(INFO) << "CreateBarrier: new barrier, name: " << name
              << ", count: " << count;

    auto p = new (&header_->handles[header_->counter_])
        MultiProcessBarrier(name, count);
    header_->counter_++;
    return p;
  }

  void ClearIPCMemory() {
    LOG(INFO) << "MultiProcessBarrierFactory->ClearIPCMemory()";
    header_->counter_ = 0;
  }

  ~MultiProcessBarrierFactory() { delete mapping_; }

  static const int kShmSize = 4 * 1024 * 1024;

 private:
  folly::MemoryMapping *mapping_;
  IPCBarrierShmRegion *header_;
};

}  // namespace recstore