#pragma once

#include <cstdint>
#include <cstring>

namespace recstore::benchmark {

inline std::uint64_t Mix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}

inline float MakeHashedValue(std::uint64_t key, int col) {
  const std::uint64_t mixed =
      Mix64(key ^ (static_cast<std::uint64_t>(col) * 0xd6e8feb86659fd93ULL));
  const std::uint32_t bits =
      0x3f800000U | (static_cast<std::uint32_t>(mixed) & 0x007fffffU);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

} // namespace recstore::benchmark
