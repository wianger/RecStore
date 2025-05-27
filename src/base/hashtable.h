#pragma once
#include <string>
#include <unordered_map>
#include <vector>

#include "base.h"
#include "base/log.h"
#include "hash.h"

namespace base {

// Note: Not thread-safe, and iteration order is unspecified.
template <class T>
class StdAutoDeleteHash {
 public:
  StdAutoDeleteHash() {}
  explicit StdAutoDeleteHash(int capacity) { map_.reserve(capacity); }

  ~StdAutoDeleteHash() { Clear(); }

  void Clear() {
    for (auto &kv : map_) delete kv.second;
    map_.clear();
  }

  auto begin() const { return map_.begin(); }
  auto begin() { return map_.begin(); }
  auto end() const { return map_.end(); }
  auto end() { return map_.end(); }

  int size() const { return static_cast<int>(map_.size()); }

  const T *GetByKey(uint64 key) const {
    auto it = map_.find(key);
    return it == map_.end() ? nullptr : it->second;
  }
  T *GetByKey(uint64 key) {
    auto it = map_.find(key);
    return it == map_.end() ? nullptr : it->second;
  }

  T *Get(const char *data, int size) { return GetByKey(CalcKey(data, size)); }

  const T *Get(uint64 sign) const {
    return Get(reinterpret_cast<const char *>(&sign), sizeof(uint64));
  }
  T *Get(uint64 sign) {
    return Get(reinterpret_cast<const char *>(&sign), sizeof(uint64));
  }
  const T *Get(const std::string &term) const {
    return Get(term.data(), term.size());
  }
  T *Get(const std::string &term) { return Get(term.data(), term.size()); }

  int InsertKey(uint64 key, T *value) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      map_[key] = value;
      return size() - 1;
    } else {
      if (it->second != value) delete it->second;
      it->second = value;
      // Find index
      int idx = 0;
      for (auto &kv : map_) {
        if (kv.first == key) return idx;
        ++idx;
      }
      return -1;
    }
  }
  int Insert(const char *data, int size, T *value) {
    return InsertKey(CalcKey(data, size), value);
  }
  int Insert(uint64 sign, T *value) {
    return Insert(reinterpret_cast<const char *>(&sign), sizeof(uint64), value);
  }
  int Insert(const std::string &term, T *value) {
    return Insert(term.data(), term.size(), value);
  }

  T *GetOrInsertByKey(uint64 key) {
    auto it = map_.find(key);
    if (it == map_.end()) {
      T *ptr = new T();
      map_[key] = ptr;
      return ptr;
    }
    return it->second;
  }

 private:
  inline static uint64 CalcKey(const char *data, int size) {
    return base::CityHash64(data, size);
  }

  std::unordered_map<uint64, T *> map_;
  DISALLOW_COPY_AND_ASSIGN(StdAutoDeleteHash);
};

}  // namespace base
