#include "flatc.h"
#include "parameters.h"

void Pack<ParameterCompressItem>::CompressAppend(std::string *output) const {
  ParameterCompressItem item;
  item.key = key;
  item.dim = dim;
  output->append(reinterpret_cast<const char *>(&item), sizeof(ParameterCompressItem));
  output->append(reinterpret_cast<const char *>(emb_data), dim * sizeof(float));
};
