#pragma once
#include "base/array.h"
#include "base/log.h"
#include <string>
#include <tuple>


#define XMH_SIMPLE_MALLOC

struct BaseKVConfig {
  int value_size = 0;
  int64_t hash_size = 0;
  size_t pool_size = 0;
  int num_threads = 0;
  int64_t capacity = 0;
  int64_t memory_pool_size = 0;
  uint64_t max_batch_keys_size= 0;
  std::string path = "";
  std::string library_file = "";
  std::string hash_name = "clht";
};

class BaseKV {
public:
  virtual ~BaseKV() {std::cout << "exit BaseKV" << std::endl;}
  explicit BaseKV(const BaseKVConfig &config){};
  virtual void Util() { std::cout << "BaseKV Util: no impl" << std::endl; return; }
  virtual void Get(const uint64_t key, std::string &value, unsigned tid) = 0;
  virtual void Put(const uint64_t key, const std::string_view &value, unsigned tid) = 0;

  virtual void BatchPut(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> &values, unsigned tid) = 0;

  virtual void BatchGet(base::ConstArray<uint64_t> keys,
                        std::vector<base::ConstArray<float>> *values, unsigned tid) = 0;

  virtual std::pair<uint64_t, uint64_t> RegisterPMAddr() const = 0;

  virtual void BulkLoad(base::ConstArray<uint64_t> keys, const void *value) { LOG(FATAL) << "not implemented"; };

  virtual void DebugInfo() const {};

  virtual void clear() { LOG(FATAL) << "not implemented"; };
};
