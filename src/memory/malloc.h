#pragma once
#include <atomic>
#include <memory>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

#include "base/base.h"
#include "epoch_manager.h"

namespace base {
class MallocApi {
public:
  virtual char* New(int memory_size)                                      = 0;
  virtual bool Free(void* memory_data)                                    = 0;
  virtual void GetMallocsAppend(std::vector<char*>* mallocs_data) const   = 0;
  virtual void GetMallocsAppend(std::vector<int64>* mallocs_offset) const = 0;
  virtual void Initialize()                                               = 0;
  virtual char* GetMallocData(int64 offset) const                         = 0;
  virtual int GetMallocSize(int64 offset) const                           = 0;
  virtual int64 GetMallocOffset(const char* data) const                   = 0;
  virtual int GetMallocSize(const char* data) const                       = 0;
  virtual int64 DataBaseOffset() const { return 0; }
  virtual std::string GetInfo() const { return ""; }
  virtual ~MallocApi() {}
  virtual uint64_t total_malloc() const                = 0;
  virtual bool Healthy() const                         = 0;
  virtual void AddMallocs4Recovery(int64_t shm_offset) = 0;

protected:
  MallocApi() {}

private:
  DISALLOW_COPY_AND_ASSIGN(MallocApi);
};

class ShmBaseRecycle {
public:
  virtual void Recycle(void* ptr) = 0;
  virtual ~ShmBaseRecycle() {}
};

class DirectRecycle : public ShmBaseRecycle {
public:
  explicit DirectRecycle(MallocApi* malloc) : malloc_(malloc) {}
  void Recycle(void* ptr) override { malloc_->Free(ptr); }

private:
  MallocApi* malloc_;
};

class DelayedRecycle : public ShmBaseRecycle {
public:
  explicit DelayedRecycle(MallocApi* malloc)
      : malloc_(malloc), delay_ts_(1000000) {}
  DelayedRecycle(MallocApi* malloc, int64 delay_ts)
      : malloc_(malloc), delay_ts_(delay_ts) {}
  ~DelayedRecycle() { Clear(); }
  void Recycle(void* ptr) override {
    char* data = reinterpret_cast<char*>(ptr);
    if (ptr) {
      recycle_ptr_.push(std::make_pair(base::GetTimestamp(), data));
    }
    int64 recyle_ts = base::GetTimestamp() - delay_ts_;
    while (recycle_ptr_.size() && recycle_ptr_.front().first < recyle_ts) {
      CHECK(malloc_->Free(recycle_ptr_.front().second))
          << recycle_ptr_.front().second;
      recycle_ptr_.pop();
    }
  }
  void Clear() {
    while (recycle_ptr_.size()) {
      CHECK(malloc_->Free(recycle_ptr_.front().second));
      recycle_ptr_.pop();
    }
  }
  int64 GetPendingNum() const { return recycle_ptr_.size(); }

  void SetDelayTs(int delay_ts) { delay_ts_ = delay_ts; }

private:
  std::queue<std::pair<int64, char*>> recycle_ptr_;
  MallocApi* malloc_;
  int64 delay_ts_;
};

class ThreadSafeDelayedRecycle : public ShmBaseRecycle {
public:
  explicit ThreadSafeDelayedRecycle(MallocApi* malloc)
      : malloc_(malloc), delay_ts_(1000000) {}
  ThreadSafeDelayedRecycle(MallocApi* malloc, int64 delay_ts)
      : malloc_(malloc), delay_ts_(delay_ts) {}
  ~ThreadSafeDelayedRecycle() override { LocalRecyclers().erase(this); }

  void Recycle(void* ptr) override { LocalRecycle().Recycle(ptr); }

  int64 GetPendingNum() const {
    auto& recyclers = LocalRecyclers();
    auto it         = recyclers.find(this);
    return it == recyclers.end() ? 0 : it->second->GetPendingNum();
  }

  void SetDelayTs(int delay_ts) {
    delay_ts_.store(delay_ts, std::memory_order_relaxed);
    auto& recyclers = LocalRecyclers();
    auto it         = recyclers.find(this);
    if (it != recyclers.end()) {
      it->second->SetDelayTs(delay_ts);
    }
  }

private:
  using RecyclerMap =
      std::unordered_map<const ThreadSafeDelayedRecycle*,
                         std::unique_ptr<DelayedRecycle>>;

  static RecyclerMap& LocalRecyclers() {
    thread_local RecyclerMap recyclers;
    return recyclers;
  }

  DelayedRecycle& LocalRecycle() const {
    auto& recyclers = LocalRecyclers();
    auto it         = recyclers.find(this);
    if (it == recyclers.end()) {
      it = recyclers
               .emplace(this,
                        std::make_unique<DelayedRecycle>(
                            malloc_, delay_ts_.load(std::memory_order_relaxed)))
               .first;
    }
    return *it->second;
  }

  MallocApi* malloc_;
  std::atomic<int64> delay_ts_;
};

class StdDelayedRecycle : public ShmBaseRecycle {
public:
  explicit StdDelayedRecycle() : delay_ts_(1000000) {}
  StdDelayedRecycle(int64 delay_ts) : delay_ts_(delay_ts) {}
  ~StdDelayedRecycle() { Clear(); }
  void Recycle(void* ptr) override {
    char* data = reinterpret_cast<char*>(ptr);
    if (ptr) {
      recycle_ptr_.push(std::make_pair(base::GetTimestamp(), data));
    }
    int64 recyle_ts = base::GetTimestamp() - delay_ts_;
    while (recycle_ptr_.size() && recycle_ptr_.front().first < recyle_ts) {
      delete recycle_ptr_.front().second;
      recycle_ptr_.pop();
    }
  }
  void Clear() {
    while (recycle_ptr_.size()) {
      delete recycle_ptr_.front().second;
      recycle_ptr_.pop();
    }
  }
  int64 GetPendingNum() const { return recycle_ptr_.size(); }

  void SetDelayTs(int delay_ts) { delay_ts_ = delay_ts; }

private:
  std::queue<std::pair<int64, char*>> recycle_ptr_;
  int64 delay_ts_;
};

class ShmEpochRecycle : public ShmBaseRecycle {
public:
  explicit ShmEpochRecycle(MallocApi* malloc) : malloc_(malloc) {
    epoch_manager_ = epoch::EpochManager::GetInstance();
  }
  ~ShmEpochRecycle() { Clear(); }
  void Recycle(void* ptr) {
    recycled_count++;
    epoch::Epoch removal_epoch = epoch_manager_->GetCurrentEpoch();
    char* data                 = reinterpret_cast<char*>(ptr);
    if (ptr) {
      recycle_ptr_.push(std::make_pair(removal_epoch, data));
    }

    if (recycled_count % 1000 == 0)
      epoch_manager_->BumpCurrentEpoch();
    while (recycle_ptr_.size() &&
           epoch_manager_->IsSafeToReclaim(recycle_ptr_.front().first)) {
      CHECK(malloc_->Free(recycle_ptr_.front().second))
          << recycle_ptr_.front().second;
      recycle_ptr_.pop();
    }
  }
  void Clear() {
    while (recycle_ptr_.size()) {
      CHECK(malloc_->Free(recycle_ptr_.front().second));
      recycle_ptr_.pop();
    }
  }
  int64 GetPendingNum() const { return recycle_ptr_.size(); }

private:
  MallocApi* malloc_;
  epoch::EpochManager* epoch_manager_;
  std::queue<std::pair<epoch::Epoch, char*>> recycle_ptr_;
  int64_t recycled_count = 0;
};

} // namespace base
