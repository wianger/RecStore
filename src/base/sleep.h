#pragma once
#include <chrono>
#include <cstdint>
#include <thread>

namespace base {
inline void SleepMs(int64_t ms) {
  std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}
}  // namespace base