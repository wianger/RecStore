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

class Atomic {
 public:
  static bool CAS(int* ptr, int old_val, int new_val) {
    return __sync_bool_compare_and_swap(ptr, old_val, new_val);
  }

  static bool CAS(void** ptr, void* old_val, void* new_val) {
    return __sync_bool_compare_and_swap(ptr, old_val, new_val);
  }

  template <typename T>
  static T load(const volatile T* obj) {
    return __atomic_load_n(obj, __ATOMIC_SEQ_CST);
  }

  template <typename T>
  static void store(volatile T* obj, T desired) {
    return __atomic_store_n(obj, desired, __ATOMIC_SEQ_CST);
  }
};

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

// class ReaderFriendlyLock {
//   std::vector<uint64_t[8]> lock_vec_;

//  public:
//   DELETE_COPY_CONSTRUCTOR_AND_ASSIGNMENT(ReaderFriendlyLock);

//   ReaderFriendlyLock(ReaderFriendlyLock&& rhs) noexcept {
//     *this = std::move(rhs);
//   }
//   ReaderFriendlyLock& operator=(ReaderFriendlyLock&& rhs) {
//     std::swap(this->lock_vec_, rhs.lock_vec_);
//     return *this;
//   }

//   ReaderFriendlyLock() : lock_vec_(util::Schedule::max_nr_threads()) {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       lock_vec_[i][0] = 0;
//       lock_vec_[i][1] = 0;
//     }
//   }

//   bool lock() {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       while (!CAS(&lock_vec_[i][0], 0, 1)) {
//       }
//     }
//     return true;
//   }

//   bool try_lock() {
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       if (!CAS(&lock_vec_[i][0], 0, 1)) {
//         for (i--; i >= 0; i--) {
//           compiler_barrier();
//           lock_vec_[i][0] = 0;
//         }
//         return false;
//       }
//     }
//     return true;
//   }

//   bool try_lock_shared() {
//     if (lock_vec_[util::Schedule::thread_id()][1]) {
//       pr_once(info, "recursive lock!");
//       return true;
//     }
//     return CAS(&lock_vec_[util::Schedule::thread_id()][0], 0, 1);
//   }

//   bool lock_shared() {
//     if (lock_vec_[util::Schedule::thread_id()][1]) {
//       pr_once(info, "recursive lock!");
//       return true;
//     }
//     while (!CAS(&lock_vec_[util::Schedule::thread_id()][0], 0, 1)) {
//     }
//     lock_vec_[util::Schedule::thread_id()][1] = 1;
//     return true;
//   }

//   void unlock() {
//     compiler_barrier();
//     for (int i = 0; i < util::Schedule::max_nr_threads(); ++i) {
//       lock_vec_[i][0] = 0;
//     }
//   }

//   void unlock_shared() {
//     compiler_barrier();
//     lock_vec_[util::Schedule::thread_id()][0] = 0;
//     lock_vec_[util::Schedule::thread_id()][1] = 0;
//   }
// };

}  // namespace base