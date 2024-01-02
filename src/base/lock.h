#pragma once
#include <atomic>

namespace base {

constexpr bool kDetectDeadLock = true;
// constexpr bool kDetectDeadLock = false;

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
            end_time - start_time).count();
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

}  // namespace base