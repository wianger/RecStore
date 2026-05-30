#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>

namespace recstore {

int FutexWait(std::atomic<uint32_t>* addr,
              uint32_t expected,
              const struct timespec* timeout);

int FutexWake(std::atomic<uint32_t>* addr, int count);

inline int FutexWakeOne(std::atomic<uint32_t>* addr) {
  return FutexWake(addr, 1);
}

inline int FutexWakeAll(std::atomic<uint32_t>* addr) {
  return FutexWake(addr, 1 << 30);
}

bool FutexWaitUntilValueChange(std::atomic<uint32_t>* addr,
                               uint32_t expected,
                               std::chrono::milliseconds timeout);

} // namespace recstore
