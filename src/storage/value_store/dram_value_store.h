#pragma once

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include "base/factory.h"
#include "memory/malloc.h"
#include "memory/memory_factory.h"
#include "storage/value_store/value_store.h"

class DramValueStore : public ValueStore {
public:
  explicit DramValueStore(const BaseKVConfig& config) {
    const auto& j = config.json_config_;
    if (!j.contains("value") || !j.at("value").contains("dram_allocator")) {
      throw std::invalid_argument(
          "DramValueStore requires value.dram_allocator");
    }
    const auto& value = j.at("value");
    if (!value.contains("path") ||
        value.at("path").get<std::string>().empty()) {
      throw std::invalid_argument(
          "DramValueStore requires non-empty value.path");
    }
    const std::string path           = value.at("path").get<std::string>();
    const auto& dram                 = value.at("dram_allocator");
    const std::string allocator_type = dram.value("type", "R2_SLAB");
    const uint64_t capacity_bytes = dram.at("capacity_bytes").get<uint64_t>();
    using MF                      = base::
        Factory<base::MallocApi, const std::string&, int64, const std::string&>;
    allocator_.reset(MF::NewInstance(
        allocator_type, path, static_cast<int64>(capacity_bytes), "DRAM"));
    if (!allocator_) {
      throw std::runtime_error("failed to create DramValueStore allocator");
    }
    allocator_->Initialize();
    recycler_ = std::make_unique<base::ThreadSafeDelayedRecycle>(
        allocator_.get(), kRecycleDelayUs);
  }

  uint64_t Alloc(size_t size) override {
    char* data = allocator_->New(static_cast<int>(size));
    if (data == nullptr) {
      return kValueHandleNone;
    }
    return EncodeOffset(allocator_->GetMallocOffset(data));
  }

  void Write(uint64_t handle, const void* data, size_t size) override {
    char* dst = Ptr(handle);
    if (dst == nullptr) {
      return;
    }
    std::memcpy(dst, data, size);
  }

  uint64_t AllocAndWrite(const void* data, size_t size) override {
    const uint64_t handle = Alloc(size);
    if (handle != kValueHandleNone) {
      Write(handle, data, size);
    }
    return handle;
  }

  size_t Read(uint64_t handle, void* out_buf, size_t buf_size) override {
    char* src = Ptr(handle);
    if (src == nullptr || out_buf == nullptr) {
      return 0;
    }
    const size_t n = std::min(
        buf_size,
        static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle))));
    std::memcpy(out_buf, src, n);
    return n;
  }

  void Free(uint64_t handle) override {
    char* data = Ptr(handle);
    if (data != nullptr) {
      allocator_->Free(data);
    }
  }

  void Retire(uint64_t handle) override {
    if (handle == kValueHandleNone) {
      return;
    }
    recycler_->Recycle(allocator_->GetMallocData(DecodeOffset(handle)));
  }

  const char* DirectPtr(uint64_t handle) const override {
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  size_t SlotCapacity(uint64_t handle) const override {
    if (handle == kValueHandleNone) {
      return 0;
    }
    return static_cast<size_t>(allocator_->GetMallocSize(DecodeOffset(handle)));
  }

  bool ReadFlatFixedRows(const uint64_t* handles,
                         size_t num_rows,
                         void* out_buf,
                         size_t row_bytes,
                         uint64_t* missing_rows) const override {
    if (handles == nullptr || out_buf == nullptr || row_bytes == 0) {
      return false;
    }
    uint64_t local_missing = 0;
    char* dst              = static_cast<char*>(out_buf);
    for (size_t row = 0; row < num_rows; ++row) {
      char* row_dst = dst + row * row_bytes;
      if (handles[row] == kValueHandleNone) {
        std::memset(row_dst, 0, row_bytes);
        ++local_missing;
        continue;
      }
      const char* src = Ptr(handles[row]);
      if (src == nullptr) {
        return false;
      }
      std::memcpy(row_dst, src, row_bytes);
    }
    if (missing_rows != nullptr) {
      *missing_rows = local_missing;
    }
    return true;
  }

  std::string GetInfo() const override { return allocator_->GetInfo(); }
  uint64_t TotalAllocCount() const { return allocator_->total_malloc(); }

private:
  static uint64_t EncodeOffset(int64 offset) {
    return static_cast<uint64_t>(offset) + 1;
  }

  static int64 DecodeOffset(uint64_t handle) {
    return static_cast<int64>(handle - 1);
  }

  char* Ptr(uint64_t handle) const {
    if (handle == kValueHandleNone) {
      return nullptr;
    }
    return allocator_->GetMallocData(DecodeOffset(handle));
  }

  std::unique_ptr<base::MallocApi> allocator_;
  std::unique_ptr<base::ThreadSafeDelayedRecycle> recycler_;
  static constexpr int64 kRecycleDelayUs = 1000;
};

FACTORY_REGISTER(
    ValueStore, DRAM_VALUE_STORE, DramValueStore, const BaseKVConfig&);
