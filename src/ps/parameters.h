#pragma once
#include "flatc.h"

#pragma pack(push, 1)
struct ParameterCompressItem {
  uint64_t key;
  int dim;

  static int GetSize(int dim) {
    return sizeof(ParameterCompressItem) + dim * sizeof(float);
  }
  const float* data() const { return embedding; }

  int byte_size() const {
    return sizeof(ParameterCompressItem) + dim * sizeof(float);
  }
  float embedding[0];  // this must be the tail
};
#pragma pack(pop)

template <>
struct Pack<ParameterCompressItem> {
  static constexpr const bool implemented = true;
  uint64_t key = 0;
  int dim = 0;
  const float* emb_data = nullptr;
  void CompressAppend(std::string* output) const;
};

using ParameterPack = Pack<ParameterCompressItem>;
using ParameterCompressor = FlatItemCompressor<ParameterCompressItem>;
using ParameterCompressReader = FlatItemCompressReader<ParameterCompressItem>;