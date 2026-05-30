#pragma once

#include <cstddef>
#include <cstdint>

#include "local_shm_protocol.h"

namespace recstore {

constexpr std::size_t kLocalShmAlignment = 64;

inline std::size_t
AlignUp(std::size_t value, std::size_t alignment = kLocalShmAlignment) {
  return (value + alignment - 1) / alignment * alignment;
}

inline std::size_t ControlBlockOffset() { return 0; }

inline std::size_t QueueHeadersOffset() {
  return AlignUp(sizeof(LocalShmControlBlock));
}

inline std::size_t TotalQueueCount(uint32_t ready_queue_count) {
  return static_cast<std::size_t>(ready_queue_count) * 2;
}

inline std::size_t QueueHeadersBytes(uint32_t ready_queue_count = 1) {
  return AlignUp(
      sizeof(LocalShmQueueHeader) * TotalQueueCount(ready_queue_count));
}

inline std::size_t QueueCellsOffset(uint32_t ready_queue_count = 1) {
  return QueueHeadersOffset() + QueueHeadersBytes(ready_queue_count);
}

inline std::size_t QueueCellsStrideBytes(uint32_t slot_count) {
  return AlignUp(
      sizeof(LocalShmQueueCell) * static_cast<std::size_t>(slot_count));
}

inline std::size_t
QueueCellsBytes(uint32_t slot_count, uint32_t ready_queue_count = 1) {
  return QueueCellsStrideBytes(slot_count) * TotalQueueCount(ready_queue_count);
}

inline std::size_t
QueueHeaderOffset(LocalQueueKind kind, uint32_t ready_queue_count = 1) {
  (void)ready_queue_count;
  return QueueHeadersOffset() +
         static_cast<std::size_t>(kind) * sizeof(LocalShmQueueHeader);
}

inline std::size_t QueueCellArrayOffset(
    uint32_t slot_count, LocalQueueKind kind, uint32_t ready_queue_count = 1) {
  return QueueCellsOffset(ready_queue_count) +
         static_cast<std::size_t>(kind) * QueueCellsStrideBytes(slot_count);
}

inline std::size_t ReadyQueueHeaderOffset(uint32_t ready_queue_id) {
  return QueueHeadersOffset() +
         sizeof(LocalShmQueueHeader) *
             static_cast<std::size_t>(1 + ready_queue_id);
}

inline std::size_t
FreeQueueHeaderOffset(uint32_t free_queue_id, uint32_t ready_queue_count = 1) {
  (void)ready_queue_count;
  return QueueHeadersOffset() +
         sizeof(LocalShmQueueHeader) * static_cast<std::size_t>(free_queue_id);
}

inline std::size_t FreeQueueCellsOffset(uint32_t slot_count,
                                        uint32_t free_queue_id,
                                        uint32_t ready_queue_count = 1) {
  return QueueCellsOffset(ready_queue_count) +
         QueueCellsStrideBytes(slot_count) *
             static_cast<std::size_t>(free_queue_id);
}

inline std::size_t
ReadyQueueHeaderOffset(uint32_t ready_queue_id, uint32_t ready_queue_count) {
  return QueueHeadersOffset() +
         sizeof(LocalShmQueueHeader) *
             static_cast<std::size_t>(ready_queue_count + ready_queue_id);
}

inline std::size_t ReadyQueueCellsOffset(uint32_t slot_count,
                                         uint32_t ready_queue_id,
                                         uint32_t ready_queue_count = 1) {
  return QueueCellsOffset(ready_queue_count) +
         QueueCellsStrideBytes(slot_count) *
             static_cast<std::size_t>(ready_queue_count + ready_queue_id);
}

inline std::size_t
SlotHeadersOffset(uint32_t slot_count, uint32_t ready_queue_count = 1) {
  return QueueCellsOffset(ready_queue_count) +
         QueueCellsBytes(slot_count, ready_queue_count);
}

inline std::size_t SlotHeadersBytes(uint32_t slot_count) {
  return AlignUp(
      sizeof(LocalShmSlotHeader) * static_cast<std::size_t>(slot_count));
}

inline std::size_t
SlotPayloadsOffset(uint32_t slot_count, uint32_t ready_queue_count = 1) {
  return SlotHeadersOffset(slot_count, ready_queue_count) +
         SlotHeadersBytes(slot_count);
}

inline std::size_t SlotPayloadOffset(
    uint32_t slot_count,
    uint32_t slot_buffer_bytes,
    uint32_t slot_id,
    uint32_t ready_queue_count = 1) {
  return SlotPayloadsOffset(slot_count, ready_queue_count) +
         static_cast<std::size_t>(slot_id) * AlignUp(slot_buffer_bytes);
}

inline std::size_t TotalRegionBytes(uint32_t slot_count,
                                    uint32_t slot_buffer_bytes,
                                    uint32_t ready_queue_count = 1) {
  return SlotPayloadsOffset(slot_count, ready_queue_count) +
         static_cast<std::size_t>(slot_count) * AlignUp(slot_buffer_bytes);
}

} // namespace recstore
