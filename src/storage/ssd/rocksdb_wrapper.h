#pragma once
#include <folly/GLog.h>
#include <folly/Portability.h>

#include <vector>

#include "naiveKVell.h"
#include "rocksdb/db.h"
#include "rocksdb/options.h"
#include "rocksdb/slice.h"

// using ROCKSDB_NAMESPACE::DB;
// using ROCKSDB_NAMESPACE::Options;
// using ROCKSDB_NAMESPACE::PinnableSlice;
// using ROCKSDB_NAMESPACE::ReadOptions;
// using ROCKSDB_NAMESPACE::Slice;
// using ROCKSDB_NAMESPACE::Status;
// using ROCKSDB_NAMESPACE::WriteBatch;
// using ROCKSDB_NAMESPACE::WriteOptions;

namespace ssdps {
template <typename KEY_T>
class RocksDBWrapper : public SsdPsInterface<KEY_T> {
 public:
  RocksDBWrapper(int VALUE_SIZE, uint64_t vector_capability)
      : VALUE_SIZE(VALUE_SIZE), vector_capability(vector_capability) {
    rocksdb::Options options;
    options.create_if_missing = true;
    options.manual_wal_flush = true;
    options.IncreaseParallelism();
    options.OptimizeForPointLookup(8);
    options.OptimizeLevelStyleCompaction();
    const std::string kDBPath = "/optane/rocksdb_example";
    rocksdb::Status s = rocksdb::DB::Open(options, kDBPath, &db_);
    CHECK(s.ok());
  }

  void BulkLoad(ConstArray<KEY_T> keys_array, const void *value) override {
    const char *value_char = (const char *)value;
    rocksdb::ColumnFamilyHandle *ch{get_or_create_column_handle_("emb")};
    for (uint64_t i = 0; i < keys_array.Size(); i++) {
      rocksdb::Status status = db_->Put(
          rocksdb::WriteOptions(), ch,
          rocksdb::Slice((char *)&keys_array[i], sizeof(KEY_T)),
          rocksdb::Slice((char *)(value_char + i * VALUE_SIZE), VALUE_SIZE));
      CHECK(status.ok());
    }
  }

  // batch get keys and save to dst with index, the index stores the slot number
  // of dst matrix (i.e. we need * VALUE_SIZE)
  void BatchGet(ConstArray<KEY_T> keys_array, ConstArray<uint64_t> index,
                void *dst) override {
    static std::vector<rocksdb::Slice> k_views;
    static std::vector<std::string> v_views;
    static std::vector<rocksdb::Status> statuses;
    static std::vector<rocksdb::ColumnFamilyHandle *> cfs;
    CHECK_EQ(keys_array.Size(), index.Size());

    int batch_size = keys_array.Size();

    k_views.clear();
    k_views.reserve(batch_size);
    v_views.resize(batch_size);
    // statuses.resize(batch_size);

    rocksdb::ColumnFamilyHandle *ch{get_or_create_column_handle_("emb")};
    cfs.resize(batch_size, ch);

    for (int i = 0; i < batch_size; i++) {
      k_views.emplace_back((char *)&keys_array[i], sizeof(KEY_T));
    }

    statuses = db_->MultiGet(rocksdb::ReadOptions(), cfs, k_views, &v_views);

    uint64_t hit_count = 0;
    for (int idx{0}; idx < batch_size; ++idx) {
      const rocksdb::Status &s{statuses[idx]};
      if (s.ok()) {
        CHECK_EQ(v_views[idx].size(), VALUE_SIZE);
        memcpy((char *)dst + index[idx] * VALUE_SIZE, v_views[idx].data(),
               v_views[idx].size());
        ++hit_count;
      } else {
        LOG(ERROR) << "key = " << keys_array[idx];
        LOG(ERROR) << "status = " << s.ToString();
        // LOG(FATAL) << "not found kv in rocksdb";
      }
    }
  }

  ~RocksDBWrapper() { delete db_; }

 private:
  inline rocksdb::ColumnFamilyHandle *get_or_create_column_handle_(
      const std::string &table_name) {
    const auto &it{column_handles_.find(table_name)};
    if (it != column_handles_.end()) {
      return it->second;
    }

    rocksdb::ColumnFamilyHandle *ch;
    db_->CreateColumnFamily(column_family_options_, table_name, &ch);
    column_handles_.emplace(table_name, ch);
    return ch;
  }

  int VALUE_SIZE;
  uint64_t vector_capability;
  rocksdb::DB *db_;
  rocksdb::ColumnFamilyOptions column_family_options_;
  std::unordered_map<std::string, rocksdb::ColumnFamilyHandle *>
      column_handles_;
};
}  // namespace ssdps