#pragma once
#include <boost/coroutine2/all.hpp>
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"
#include "storage/hybrid/pointer.h"

using boost::coroutines2::coroutine;

struct IndexConfig {
  json json_config_;  // add your custom config in this field
};

class Index {
 public:
  virtual ~Index() { std::cout << "exit Index" << std::endl; }

  explicit Index(const IndexConfig &config){};

  virtual void Util() {
    std::cout << "Index Util: no impl" << std::endl;
    return;
  }
  virtual void Get(const uint64_t key, uint64_t &pointer, unsigned tid) = 0;
  virtual void Put(const uint64_t key, uint64_t pointer, unsigned tid) = 0;

  virtual void BatchPut(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        uint64_t* pointers,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  };

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        uint64_t* pointers,
                        unsigned tid) = 0;

  virtual void BatchGet(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        uint64_t* pointers,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual void DebugInfo() const {}

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) {
    LOG(FATAL) << "not implemented";
  };

  virtual void LoadFakeData(int64_t key_capacity, int value_size) {
    std::vector<uint64_t> keys;
    uint64_t *values = new uint64_t[value_size / sizeof(uint64_t) * key_capacity];
    keys.reserve(key_capacity);
    for (int64_t i = 0; i < key_capacity; i++) {
      keys.push_back(i);
      *(values + i ) = i;  
      uint64_t ptr_value = *reinterpret_cast<const uint64_t *>(values + i );
    }
    this->BulkLoad(base::ConstArray<uint64_t>(keys), values);
    delete[] values;
  };

  virtual void clear() { LOG(FATAL) << "not implemented"; };

  virtual std::string RetrieveValue(uint64_t raw_value) { 
    UnifiedPointer p = UnifiedPointer::FromRaw(raw_value);
    std::cout << "Raw pointer info: " << p.toString() << std::endl;

    switch (p.type()) { // TODO: retrieve value from different tier
      case UnifiedPointer::Type::Memory: {
        std::cout << "Memory Pointer: " << p.asMemoryPointer() << std::endl;
        break;
      }
      case UnifiedPointer::Type::Disk:
        std::cout << "Disk PageID: " << p.asDiskPageId() << std::endl;
        break;
      case UnifiedPointer::Type::PMem:
        std::cout << "PMem Offset: " << p.asPMemOffset() << std::endl;
        break;
      case UnifiedPointer::Type::Invalid:
      default:
        std::cout << "Invalid pointer type." << std::endl;
        break;
    }

    return "";
  }

 protected:
};
