#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include "base/log.h"

namespace petps {

inline constexpr std::uint32_t kPutPayloadMagic      = 0x50545053;
inline constexpr std::uint16_t kPutProtocolVersionV1 = 1;

struct PutPayloadHeader {
  std::uint32_t magic;
  std::uint16_t version;
  std::uint16_t reserved;
  std::uint32_t key_count;
  std::uint32_t embedding_dim;
};

struct DecodedPutPayload {
  std::uint32_t embedding_dim = 0;
  std::vector<std::uint64_t> keys;
  std::vector<float> values;
};

inline std::size_t
FixedSlotResponseBytes(std::size_t key_count, std::size_t value_size_bytes) {
  return key_count * value_size_bytes + sizeof(std::int32_t);
}

inline std::size_t
PutPayloadBytes(std::size_t key_count, std::size_t value_size_bytes) {
  return key_count * (sizeof(std::uint64_t) + value_size_bytes);
}

inline std::string
EncodePutPayload(const std::vector<std::uint64_t>& keys,
                 const std::vector<std::vector<float>>& values,
                 std::string* error = nullptr) {
  if (keys.empty()) {
    return {};
  }

  if (keys.size() != values.size()) {
    if (error != nullptr) {
      *error = "keys and values size mismatch";
    }
    LOG(ERROR) << "EncodePutPayload: keys.size()=" << keys.size()
               << " != values.size()=" << values.size();
    return {};
  }

  const std::size_t embedding_dim = values.front().size();
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (values[i].size() != embedding_dim) {
      if (error != nullptr) {
        *error = "inconsistent embedding dimension";
      }
      LOG(ERROR) << "EncodePutPayload: values[" << i
                 << "].size()=" << values[i].size()
                 << " != expected embedding_dim=" << embedding_dim;
      return {};
    }
  }

  PutPayloadHeader header{
      kPutPayloadMagic,
      kPutProtocolVersionV1,
      0,
      static_cast<std::uint32_t>(keys.size()),
      static_cast<std::uint32_t>(embedding_dim),
  };

  std::string payload;
  payload.resize(
      sizeof(PutPayloadHeader) + keys.size() * sizeof(std::uint64_t) +
      keys.size() * embedding_dim * sizeof(float));

  char* cursor = payload.data();
  std::memcpy(cursor, &header, sizeof(header));
  cursor += sizeof(header);

  std::memcpy(cursor, keys.data(), keys.size() * sizeof(std::uint64_t));
  cursor += keys.size() * sizeof(std::uint64_t);

  for (const auto& row : values) {
    std::memcpy(cursor, row.data(), embedding_dim * sizeof(float));
    cursor += embedding_dim * sizeof(float);
  }

  return payload;
}

inline bool DecodePutPayload(
    std::string_view payload, DecodedPutPayload* decoded, std::string* error) {
  if (decoded == nullptr) {
    if (error != nullptr) {
      *error = "decoded is null";
    }
    return false;
  }
  if (payload.size() < sizeof(PutPayloadHeader)) {
    if (error != nullptr) {
      *error = "payload smaller than header";
    }
    return false;
  }

  PutPayloadHeader header{};
  std::memcpy(&header, payload.data(), sizeof(header));
  if (header.magic != kPutPayloadMagic) {
    if (error != nullptr) {
      *error = "bad payload magic";
    }
    return false;
  }
  if (header.version != kPutProtocolVersionV1) {
    if (error != nullptr) {
      *error = "bad protocol version";
    }
    return false;
  }

  const std::size_t expected_bytes =
      sizeof(PutPayloadHeader) + header.key_count * sizeof(std::uint64_t) +
      static_cast<std::size_t>(header.key_count) * header.embedding_dim *
          sizeof(float);
  if (payload.size() != expected_bytes) {
    if (error != nullptr) {
      *error = "payload byte size mismatch";
    }
    return false;
  }

  decoded->embedding_dim = header.embedding_dim;
  decoded->keys.resize(header.key_count);
  decoded->values.resize(
      static_cast<std::size_t>(header.key_count) * header.embedding_dim);

  const char* cursor = payload.data() + sizeof(PutPayloadHeader);
  std::memcpy(
      decoded->keys.data(), cursor, header.key_count * sizeof(std::uint64_t));
  cursor += header.key_count * sizeof(std::uint64_t);
  std::memcpy(
      decoded->values.data(), cursor, decoded->values.size() * sizeof(float));

  return true;
}

} // namespace petps
