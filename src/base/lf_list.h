#ifndef __LF_LIST__
#define __LF_LIST__

#include <atomic>
#include <vector>
#include <glog/logging.h>

namespace base {

class LFList{
private:
  std::atomic<int> free_head;
  std::atomic<int> free_tail;
  int *data_;
  int list_capacity_;
public:
  int Capacity() const {
    return list_capacity_;
  }

  void InsertFreeList(std::vector<int> &keys) {
    int free_size = free_tail.load() - free_head.load();
    if (free_size < 0) { free_size += list_capacity_; }
    if(free_size + keys.size() >= list_capacity_){
      CHECK(0) << "free list is full";
    }
    int stop_pos = (free_tail + keys.size()) % list_capacity_;
    int j = 0;
    for (int i = free_tail; i != stop_pos; i = (i + 1) % list_capacity_) {
      data_[i] = keys[j++];
    }
    free_tail.store(stop_pos);
  }

  std::pair<int, int> TryPop(int cnt) {
    // return {-1, -1};
    while (cnt > 0) {
      int head_now = free_head.load();
      int res = free_tail.load() - head_now;
      if(res < 0){
        res += list_capacity_;
      }
      if (res >= cnt) {
        int stop_pos = (head_now + cnt) % list_capacity_;
        if (free_head.compare_exchange_strong(head_now, stop_pos)) {
          return std::make_pair(head_now, stop_pos);
        }
      }
      cnt /= 2;
    }
    return std::make_pair(-1, -1);
  }

  LFList(int capacity) {
    list_capacity_ = capacity;
    data_ = new int[capacity];
    free_head.store(0);
    free_tail.store(0);
  }

  void clear(){
    free_head.store(0);
    free_tail.store(0);
  }

  const int& operator[](int index) const {
    return data_[index];
  }

  int &operator[](int index) {
    return data_[index];
  }

};

}

#endif // __LF_LIST__