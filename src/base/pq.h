#pragma once
#include <iostream>
#include <unordered_map>
#include <vector>

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
    data_.push_back(value);
    index_map_[value] = data_.size() - 1;
    heapifyUp(data_.size() - 1);
  }

  void pop() {
    index_map_.erase(data_.front());
    std::swap(data_.front(), data_.back());
    index_map_[data_.front()] = 0;
    data_.pop_back();
    heapifyDown(0);
  }

  const T& top() const { return data_.front(); }

  size_t size() const { return data_.size(); }

  void adjustPriority(const T& oldValue) {
    auto it = index_map_.find(oldValue);
    if (it != index_map_.end()) {
      size_t index = it->second;
      //   data_[index] = newValue;
      //   index_map_.erase(oldValue);
      //   index_map_[newValue] = index;

      auto& newValue = oldValue;
      // 如果元素上升
      if (index > 0 && compare(newValue, data_[(index - 1) / 2])) {
        heapifyUp(index);
      }
      // 如果元素下降
      else {
        heapifyDown(index);
      }
    } else {
      LOG(FATAL) << "adjustPriority error:"
                 << " not found";
    }
  }

  bool empty() const { return data_.empty(); }

 private:
  std::vector<T> data_;
  std::unordered_map<T, size_t> index_map_;
  Compare compare;

  void heapifyUp(size_t index) {
    while (index > 0 && compare(data_[index], data_[(index - 1) / 2])) {
      std::swap(data_[index], data_[(index - 1) / 2]);
      index = (index - 1) / 2;
    }
  }

  void heapifyDown(size_t index) {
    size_t size = data_.size();
    while (2 * index + 1 < size) {
      size_t leftChild = 2 * index + 1;
      size_t rightChild = 2 * index + 2;
      size_t smallestChild = leftChild;

      if (rightChild < size && compare(data_[rightChild], data_[leftChild])) {
        smallestChild = rightChild;
      }

      if (compare(data_[smallestChild], data_[index])) {
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