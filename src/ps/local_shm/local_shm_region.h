#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "base/log.h"
#include "local_shm_layout.h"

namespace recstore {

class LocalShmRegion {
public:
  LocalShmRegion() = default;
  ~LocalShmRegion();

  LocalShmRegion(const LocalShmRegion&)            = delete;
  LocalShmRegion& operator=(const LocalShmRegion&) = delete;

  bool Create(const std::string& region_name,
              uint32_t slot_count,
              uint32_t slot_buffer_bytes,
              uint32_t ready_queue_count = 1);
  bool Attach(const std::string& region_name,
              uint32_t expected_slot_count        = 0,
              uint32_t expected_slot_buffer_bytes = 0,
              uint32_t expected_ready_queue_count = 0);
  void Close();

  bool IsOpen() const { return base_ != nullptr; }
  const std::string& region_name() const { return region_name_; }
  uint32_t slot_count() const { return slot_count_; }
  uint32_t slot_buffer_bytes() const { return slot_buffer_bytes_; }
  uint32_t ready_queue_count() const { return ready_queue_count_; }
  std::size_t mapped_size() const { return mapped_size_; }

  LocalShmControlBlock* control();
  const LocalShmControlBlock* control() const;
  LocalShmQueueHeader* queue_header(LocalQueueKind kind);
  const LocalShmQueueHeader* queue_header(LocalQueueKind kind) const;
  LocalShmQueueCell* queue_cells(LocalQueueKind kind);
  const LocalShmQueueCell* queue_cells(LocalQueueKind kind) const;
  LocalShmQueueHeader* free_queue_header(uint32_t free_queue_id);
  const LocalShmQueueHeader* free_queue_header(uint32_t free_queue_id) const;
  LocalShmQueueCell* free_queue_cells(uint32_t free_queue_id);
  const LocalShmQueueCell* free_queue_cells(uint32_t free_queue_id) const;
  LocalShmQueueHeader* ready_queue_header(uint32_t ready_queue_id);
  const LocalShmQueueHeader* ready_queue_header(uint32_t ready_queue_id) const;
  LocalShmQueueCell* ready_queue_cells(uint32_t ready_queue_id);
  const LocalShmQueueCell* ready_queue_cells(uint32_t ready_queue_id) const;
  LocalShmSlotHeader* slot_header(uint32_t slot_id);
  const LocalShmSlotHeader* slot_header(uint32_t slot_id) const;
  uint8_t* slot_payload(uint32_t slot_id);
  const uint8_t* slot_payload(uint32_t slot_id) const;

private:
  bool MapRegion(int fd, std::size_t mapped_size);
  bool ValidateGeometry(uint32_t expected_slot_count,
                        uint32_t expected_slot_buffer_bytes,
                        uint32_t expected_ready_queue_count) const;

private:
  std::string region_name_;
  int fd_                     = -1;
  void* base_                 = nullptr;
  std::size_t mapped_size_    = 0;
  uint32_t slot_count_        = 0;
  uint32_t slot_buffer_bytes_ = 0;
  uint32_t ready_queue_count_ = 0;
};

} // namespace recstore
