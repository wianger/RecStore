#pragma once

#include "base/tensor.h"

namespace recstore {
void EmbRead(const base::RecTensor& keys, base::RecTensor& values);
void EmbUpdate(const base::RecTensor& keys, const base::RecTensor& grads);
} // namespace recstore
