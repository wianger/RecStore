#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <thread>

#include "ps/local_shm/local_shm_futex.h"

namespace recstore {
namespace {

TEST(LocalShmFutexTest, WaitUntilValueChangeWakesAfterUpdate) {
  std::atomic<uint32_t> word{0};
  bool woke = false;

  std::thread waiter([&]() {
    woke = FutexWaitUntilValueChange(&word, 0, std::chrono::milliseconds(200));
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  word.store(1, std::memory_order_release);
  EXPECT_GE(FutexWakeAll(&word), 0);

  waiter.join();
  EXPECT_TRUE(woke);
}

TEST(LocalShmFutexTest, WakeOneWakesSingleWaiter) {
  std::atomic<uint32_t> word{0};
  std::atomic<int> woke_count{0};

  std::thread waiter_a([&]() {
    if (FutexWaitUntilValueChange(&word, 0, std::chrono::milliseconds(200))) {
      woke_count.fetch_add(1, std::memory_order_relaxed);
    }
  });
  std::thread waiter_b([&]() {
    if (FutexWaitUntilValueChange(&word, 0, std::chrono::milliseconds(200))) {
      woke_count.fetch_add(1, std::memory_order_relaxed);
    }
  });

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  EXPECT_GE(FutexWakeOne(&word), 0);
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const int after_wake_one = woke_count.load(std::memory_order_relaxed);
  EXPECT_LE(after_wake_one, 1);

  word.store(1, std::memory_order_release);
  EXPECT_GE(FutexWakeAll(&word), 0);

  waiter_a.join();
  waiter_b.join();
  EXPECT_EQ(woke_count.load(std::memory_order_relaxed), 2);
}

TEST(LocalShmFutexTest, WaitUntilValueChangeTimesOut) {
  std::atomic<uint32_t> word{7};
  const auto start = std::chrono::steady_clock::now();
  const bool woke =
      FutexWaitUntilValueChange(&word, 7, std::chrono::milliseconds(20));
  const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - start);

  EXPECT_FALSE(woke);
  EXPECT_GE(elapsed.count(), 15);
}

} // namespace
} // namespace recstore
