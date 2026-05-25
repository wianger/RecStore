#pragma once

#include <memory>
#include <string>
#include <vector>

#include "memory/malloc.h"
#include "memory/shm_file.h"
#include "base/factory.h"
#include "memory/allocators/r2/allocator_master.hh"

namespace base {
class R2alloc : public MallocApi {
public:
  // filename: 文件名, 内存大小
  ~R2alloc() override { LOG(INFO) << "~R2alloc() called"; }
  R2alloc(const std::string& filename,
          int64 memory_size,
          std::string medium = "DRAM")
      : memory_size_(memory_size) {
    LOG(INFO) << "filename:" << filename << ", memory_size:" << memory_size
              << ", medium:" << medium;
    bool file_exists = base::file_util::PathExists(filename);
    shm_file_ = ShmFile::New(ShmFile::ConfigForMedium(
        medium, filename, memory_size));
    if (!shm_file_) {
      file_exists = false;
      CHECK(base::file_util::Delete(filename, false));
      shm_file_ = ShmFile::New(ShmFile::ConfigForMedium(
          medium, filename, memory_size));
      CHECK(shm_file_)
          << filename << " " << memory_size;
    }

    data_ = shm_file_->Data();
    LOG(INFO) << "init data_ addr:" << static_cast<void*>(data_);
    // Use the full mapped budget for the allocator. Using half here can make
    // concurrent arena creation fail under moderate write concurrency.
    R2AllocMaster.init(data_, memory_size);
    healthy_used_ = memory_size;
    load_success_ = file_exists;
    Initialize();
    if (!file_exists) {
      LOG(INFO) << "R2malloc: first initialization.";
    } else {
      LOG(INFO) << "R2malloc: Recovery from shutdown.";
    }
  }

  int64 DataBaseOffset() const { return 0; }

  char* New(int memory_size) {
    r2::Allocator* alloc = R2AllocMaster.get_thread_allocator();
    if (!alloc) {
      return nullptr;
    }
    void* p = alloc->alloc(memory_size);
    return static_cast<char*>(p);
  }

  bool Free(void* memory_data) {
    if (!memory_data)
      return true;
    r2::Allocator* alloc = R2AllocMaster.get_thread_allocator();
    if (!alloc)
      return false;
    alloc->dealloc(memory_data);
    return true;
  }

  bool load_success() const { return load_success_; }
  void GetMallocsAppend(std::vector<char*>* mallocs_data) const {
    // TODO
    LOG(INFO) << "NOT IMPLEMENT!";
    return;
  }
  void GetMallocsAppend(std::vector<int64>* mallocs_offset) const {
    // TODO
    LOG(INFO) << " NOT IMPLEMENT!";
    return;
  }
  std::string GetInfo() const {
    // TODO
    std::string info;
    info.append(base::StringPrintf("total_malloc: %ld\n", total_malloc_));
    return info;
  }
  void Initialize() {
    total_used_   = 0;
    total_malloc_ = 0;
  }
  // 一共使用了多少 8 字节的 block
  int64 total_used() const { return total_used_; }
  // 一共分配了多少块内存, 和 GetUsedBlockAppend 对应
  uint64 total_malloc() const { return total_malloc_; }
  bool Healthy() const { return total_used_ <= healthy_used_; }

  char* GetMallocData(int64 offset) const {
    if (offset > memory_size_ || offset < 0) {
      LOG(INFO) << "GetMallocData offset invalid:" << offset;
      return NULL;
    }
    return data_ + offset;
  }

  int GetMallocSize(int64 offset) const {
    char* data = GetMallocData(offset);
    if (data == NULL)
      return -1;
    else
      return GetMallocSize(data);
  }

  int64 GetMallocOffset(const char* data) const {
    int64 offset = data - data_;
    if (offset < 0 || offset > memory_size_ || (offset & 7) != 0) {
      LOG(INFO) << "GetMallocOffset offset invalid:" << offset;
      return -1;
    }
    return offset;
  }

  int GetMallocSize(const char* data) const {
    int malloc_size = jemalloc_usable_size((void*)data);
    return malloc_size;
  }

  void AddMallocs4Recovery(int64_t shm_offset) {
    LOG(INFO) << " NOT IMPLEMENT!";
    return;
  }

private:
  std::unique_ptr<ShmFile> shm_file_;
  char* data_;
  int64 memory_size_;
  int64 total_used_;
  int64 total_malloc_;
  int64 healthy_used_;
  bool load_success_;

  DISALLOW_COPY_AND_ASSIGN(R2alloc);
  r2::AllocatorMaster R2AllocMaster;
};
FACTORY_REGISTER(MallocApi,
                 R2ShmMalloc, // 注册名（通常同类名）
                 R2alloc,     // 具体类型
                 const std::string&,
                 int64,
                 const std::string&);
FACTORY_REGISTER(
    MallocApi, R2_SLAB, R2alloc, const std::string&, int64, const std::string&);
} // namespace base
