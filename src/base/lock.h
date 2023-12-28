#pragma once
#include <atomic>

namespace base {
class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void Lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
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
class LockGurad {
  T& lock_;

 public:
  LockGurad(T& lock) : lock_(lock) { lock_.Lock(); }
  ~LockGurad() { lock_.Unlock(); }
};

}  // namespace base