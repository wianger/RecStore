#pragma once

namespace recstore {

class MathUtil {
 public:
  static inline int round_up_to(int num, int factor) {
    return num + factor - 1 - (num + factor - 1) % factor;
  }
};

}  // namespace recstore