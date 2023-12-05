/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

#include "base/cu_utils.cuh"

namespace nv {

inline int get_dev(const void* ptr) {
  cudaPointerAttributes attr;
  CUDA_CHECK(cudaPointerGetAttributes(&attr, ptr));
  int dev = -1;

#if CUDART_VERSION >= 10000
  if (attr.type == cudaMemoryTypeDevice)
#else
  if (attr.memoryType == cudaMemoryTypeDevice)
#endif
  {
    dev = attr.device;
  }
  return dev;
}

inline void switch_to_dev(const void* ptr) {
  int dev = get_dev(ptr);
  if (dev >= 0) {
    CUDA_CHECK(cudaSetDevice(dev));
  }
}

}  // namespace nv
