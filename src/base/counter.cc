#include "counter.h"

#include <map>

#include <folly/String.h>

namespace base {

static std::map<std::string, Counter *> counters;
static base::Lock counters_lock;

void Counter::Inc(int64 count) {
  int64 now = base::GetTimestamp();
  second_count_.Update(now, count_, 1000000L);
  minute_count_.Update(now, count_, 1000000L * 60);
  hour_count_.Update(now, count_, 1000000L * 3600);
  day_count_.Update(now, count_, 1000000L * 3600L * 24L);
  count_ += count;
}

std::string Counter::Display() const {
  return folly::stringPrintf("[%s] qps: %ld, minute: %ld, hour: %ld, day: %ld, total: %ld",
                            name_.c_str(),
                            second_count_.GetCount(1000000L),
                            minute_count_.GetCount(1000000L * 60),
                            hour_count_.GetCount(1000000L * 3600),
                            day_count_.GetCount(1000000L * 3600L * 24L),
                            count_);
}

}  // namespace base
