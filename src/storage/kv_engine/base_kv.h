#pragma once
#include <boost/coroutine2/all.hpp>
#include <string>
#include <tuple>

#include "base/array.h"
#include "base/json.h"
#include "base/log.h"

using boost::coroutines2::coroutine;

#define XMH_SIMPLE_MALLOC

struct BaseKVConfig {
  int num_threads_ = 0;
  json json_config_;  // add your custom config in this field
};

class BaseKV {
 public:
  virtual ~BaseKV() { std::cout << "exit BaseKV" << std::endl; }
  explicit BaseKV(const BaseKVConfig &config){};
  virtual void Util() {
    std::cout << "BaseKV Util: no impl" << std::endl;
    return;
  }
  virtual void Get(const uint64_t key, std::string &value, unsigned tid) = 0;
  virtual void Put(const uint64_t key, const std::string_view &value,
                   unsigned tid) = 0;

  virtual void BatchPut(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  };

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) = 0;

  virtual void BatchGet(coroutine<void>::push_type &sink,
                        base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values,
                        unsigned tid) {
    LOG(FATAL) << "not implemented";
  }

  virtual void DebugInfo() const {}

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) {
    LOG(FATAL) << "not implemented";
  };

  virtual void clear() { LOG(FATAL) << "not implemented"; };
};
