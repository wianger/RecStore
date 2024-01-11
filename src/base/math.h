#pragma once

namespace recstore {

class MathUtil {
 public:
  static inline int round_up_to(int num, int factor) {
    return num + factor - 1 - (num + factor - 1) % factor;
  }

  static inline int64_t round_up_to_int64(int64_t num, int factor) {
    return num + factor - 1 - (num + factor - 1) % factor;
  }
};

}  // namespace recstore