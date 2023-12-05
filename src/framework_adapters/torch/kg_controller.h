#pragma once
#include <torch/extension.h>

namespace recstore {
void RegisterKGCacheController(torch::Library &m);
}