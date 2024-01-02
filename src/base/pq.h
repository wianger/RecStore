#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>

#include "base/lock.h"

// #define XMH_DEBUG

namespace base {
template <typename T, typename Compare = std::less<T>>
class CustomPriorityQueue {
 public:
  CustomPriorityQueue(int reserve_size = 0) {
    if (reserve_size > 0) {
      data_.reserve(reserve_size);
      index_map_.reserve(reserve_size);
    }
  }

  void push(const T& value) {
    base::LockGuard _(lock_);
    push_inner(value);
  }

  T pop() {
    base::LockGuard _(lock_);
    T ret;
    ret = data_.front();
    index_map_.erase(data_.front());
    if (data_.size() == 1) {
      data_.pop_back();
#ifdef XMH_DEBUG
      CheckConsistency("pop1");
#endif
      return ret;
    }

    std::swap(data_.front(), data_.back());
    index_map_[data_.front()] = 0;
    data_.pop_back();
    heapifyDown(0);
#ifdef XMH_DEBUG
    CheckConsistency("pop2");
    // LOG(WARNING) << "pop " << ret->GetID();
#endif
    return ret;
  }

  T pop_x(const T& value) {
    base::LockGuard _(lock_);
    T ret = value;
    int64_t old_position = index_map_[value];
    index_map_.erase(value);

    if (data_.size() == 1) {
      data_.pop_back();
#ifdef XMH_DEBUG
      CheckConsistency("pop_x_1");
#endif
      return ret;
    }

    if (old_position == data_.size() - 1) {
      data_.pop_back();
    } else {
      std::swap(data_[old_position], data_.back());
      data_.pop_back();
      T newValue = data_[old_position];
      index_map_[newValue] = old_position;
      adjustPriority(newValue);
    }

#ifdef XMH_DEBUG
    CheckConsistency("pop_x");
#endif
    return ret;
  }

  void PushOrUpdate(const T& value) {
    base::LockGuard _(lock_);
    if (index_map_.find(value) == index_map_.end()) {
      push_inner(value);
    } else {
      adjustPriority(value);
    }
  }

  T top() const {
    base::LockGuard _(lock_);
    return data_.front();
  }

  size_t size() const {
    base::LockGuard _(lock_);
    return data_.size();
  }

  bool empty() const {
    base::LockGuard _(lock_);
    return data_.empty();
  }

  std::string ToString() const {
    // base::LockGuard _(lock_);
    std::stringstream ss;
    ss << "CustomPriorityQueue:\n";
    if (data_.empty()) {
      ss << "\t\t"
         << "empty\n";
      return ss.str();
    }

    for (auto each : data_) {
      ss << "\t\t" << each->ToString() << "\n";
    }
    return ss.str();
  }

  void ForDebug(const std::string& head) {
    base::LockGuard _(lock_);

    // for (auto each : data_) {
    //   if (each->GetID() == 1718) {
    //     LOG(INFO) << head << " find 1718 " << each->ToString() << ".\n top is
    //     "
    //               << top()->ToString();
    //     CheckConsistency();
    //     return;
    //   }
    // }
    CheckConsistency();
  }

  void CheckConsistency(const std::string& hint = "") {
    lock_.AssertLockHold();

    CHECK_EQ(data_.size(), index_map_.size()) << hint;
    for (int i = 0; i < data_.size(); i++) {
      CHECK_EQ(index_map_[data_[i]], i) << hint;
    }
    int64_t heap_size = data_.size();
    for (int64_t index = 0; index < heap_size; index++) {
      size_t leftChild = 2 * index + 1;
      size_t rightChild = 2 * index + 2;
      if (leftChild < heap_size) {
        CHECK(!compare(data_[index], data_[leftChild])) << hint;
      }
      if (rightChild < heap_size) {
        CHECK(!compare(data_[index], data_[rightChild])) << hint;
      }
    }
  }

 private:
  std::vector<T> data_;
  std::unordered_map<T, size_t> index_map_;
  Compare compare;
  mutable base::SpinLock lock_;

  void push_inner(const T& value) {
    data_.push_back(value);
    index_map_[value] = data_.size() - 1;
    heapifyUp(data_.size() - 1);
#ifdef XMH_DEBUG
    CheckConsistency("push_inner");
    // LOG(WARNING) << "push " << value->GetID();
#endif
  }

  void adjustPriority(const T& oldValue) {
    auto it = index_map_.find(oldValue);
    if (it != index_map_.end()) {
      size_t index = it->second;
      auto& newValue = oldValue;
      if (index > 0 && compare(data_[(index - 1) / 2], newValue)) {
        heapifyUp(index);
      } else {
        heapifyDown(index);
      }
    } else {
      LOG(FATAL) << "adjustPriority error:"
                 << " not found";
    }

#ifdef XMH_DEBUG
    CheckConsistency("adjustPriority");
    // LOG(WARNING) << "adjustPriority " << oldValue->GetID();
    //  << "| pq: " << ToString();
#endif
  }

  void heapifyUp(size_t index) {
    while (index > 0 && compare(data_[(index - 1) / 2], data_[index])) {
      std::swap(data_[index], data_[(index - 1) / 2]);
      index_map_[data_[index]] = index;
      index_map_[data_[(index - 1) / 2]] = (index - 1) / 2;

      index = (index - 1) / 2;
    }
  }

  void heapifyDown(size_t index) {
    size_t size = data_.size();
    while (2 * index + 1 < size) {
      size_t leftChild = 2 * index + 1;
      size_t rightChild = 2 * index + 2;
      size_t smallestChild = leftChild;

      if (rightChild < size && compare(data_[leftChild], data_[rightChild])) {
        smallestChild = rightChild;
      }

      if (compare(data_[index], data_[smallestChild])) {
        std::swap(data_[index], data_[smallestChild]);
        index_map_[data_[index]] = index;
        index_map_[data_[smallestChild]] = smallestChild;
        index = smallestChild;
      } else {
        break;
      }
    }
  }
};
}  // namespace base