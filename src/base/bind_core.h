#include <pthread.h>
#include <sched.h>

#include <atomic>
#include <cstring>
#include <iostream>

#include "base/log.h"

namespace base {
constexpr int kMaxLogicCoreCnt = 100;
constexpr int kMaxSocketCnt = 2;
extern int global_socket_id;

inline void bind_core(int n) {
  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(n, &mask);
  if (sched_setaffinity(0, sizeof(mask), &mask) == -1) {
    std::cout << "Could not set CPU affinity" << std::endl;
  }
}

void auto_bind_core(void);

}  // namespace base