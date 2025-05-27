#pragma once

#include <string>
#include <vector>
#include "base.h"

namespace base {

class Counter {
 public:
  struct IntervalCount {
    int64 timestamp;
    int64 start_count;
    int64 end_count;
    void Update(int64 now, int64 current_count, int64 interval) {
      if (now - timestamp > interval) {
        if (now - timestamp > 2 * interval) {
          timestamp += ((now - timestamp) / interval) * interval;
        } else {
          timestamp += interval;
        }
        start_count = end_count;
        end_count = current_count;
      }
    }
    int64 GetCount() const { return end_count - start_count; }
    int64 GetCount(int64 interval) const {
      if (base::GetTimestamp() - timestamp < interval) {
        return end_count - start_count;
      }
      return 0;
    }
    IntervalCount() : timestamp(0), start_count(0), end_count(0) {}
  };
  Counter() : count_(0) {}
  explicit Counter(const std::string &name) : name_(name), count_(0) {}
  explicit Counter(const char *name) : name_(name), count_(0) {}
  void Inc(int64 count);
  void SetName(const std::string &name) { name_ = name; }
  void ThreadSafeInc(int64 count) {
    base::AutoLock lock(lock_);
    Inc(count);
  }
  int64 GetCount() const { return count_; }
  const std::string &GetName() const { return name_; }
  std::string Display() const;
  int64 GetQPS() const { return second_count_.GetCount(1000000L); }
  int64 GetMinuteCount() const { return minute_count_.GetCount(1000000L * 60); }

 private:
  std::string name_;
  int64 count_;
  IntervalCount second_count_;
  IntervalCount minute_count_;
  IntervalCount hour_count_;
  IntervalCount day_count_;

  base::Lock lock_;

  DISALLOW_COPY_AND_ASSIGN(Counter);
};

}  // namespace base
