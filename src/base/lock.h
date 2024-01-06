#pragma once
#include <atomic>
#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <thread>

#include "base/log.h"

namespace base {

// constexpr bool kDetectDeadLock = true;
constexpr bool kDetectDeadLock = false;

class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void Lock() {
    std::chrono::time_point<std::chrono::steady_clock> start_time;
    if (kDetectDeadLock) start_time = std::chrono::steady_clock::now();

    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
      if (kDetectDeadLock) {
        auto end_time = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                            end_time - start_time)
                            .count();
        if (duration > 1) LOG(FATAL) << "may be dead lock";
      }
    }
  }
  void Unlock() { locked.clear(std::memory_order_release); }

  void AssertLockHold() {
    assert(locked.test_and_set(std::memory_order_acquire));
  }
};

class PlaceboLock {
 public:
  void Lock() { ; }
  void Unlock() { ; }
};

template <class T>
class LockGuard {
  T& lock_;

 public:
  LockGuard(T& lock) : lock_(lock) { lock_.Lock(); }
  ~LockGuard() { lock_.Unlock(); }
};

class Barrier {
 public:
  explicit Barrier(int count) : count_(count), bar_(0) {}

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

 private:
  int count_;
  std::atomic_int bar_;
  std::atomic_int passed_ = 0;
};

}  // namespace base