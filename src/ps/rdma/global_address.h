#pragma once

#include <cstdint>

namespace petps {

struct GlobalAddress {
  std::uint64_t nodeID : 16;
  std::uint64_t offset : 48;

  static GlobalAddress Null() { return GlobalAddress{0, 0}; }
} __attribute__((packed));

static_assert(sizeof(GlobalAddress) == sizeof(std::uint64_t),
              "GlobalAddress must remain 64-bit packed");

inline GlobalAddress GADD(const GlobalAddress& address, int offset) {
  GlobalAddress result = address;
  result.offset += static_cast<std::uint64_t>(offset);
  return result;
}

inline bool operator==(const GlobalAddress& lhs, const GlobalAddress& rhs) {
  return lhs.nodeID == rhs.nodeID && lhs.offset == rhs.offset;
}

inline bool operator!=(const GlobalAddress& lhs, const GlobalAddress& rhs) {
  return !(lhs == rhs);
}

} // namespace petps
