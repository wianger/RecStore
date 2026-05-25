#pragma once

#include <functional>
#include <set>
#include <stdexcept>
#include <string>

#include "base_kv.h"

namespace base {

inline bool IsDevShmPath(const std::string& path) {
  static const std::string kPrefix = "/dev/shm";
  return path.rfind(kPrefix, 0) == 0;
}

inline void ValidateDramPathPolicy(const json& node,
                                   const char* field_name,
                                   const char* component_name) {
  if (!node.contains(field_name)) {
    return;
  }
  const std::string path = node.at(field_name).get<std::string>();
  if (path.empty()) {
    // Empty path means anonymous mmap mode for DRAM.
    return;
  }
  if (!IsDevShmPath(path)) {
    throw std::invalid_argument(std::string(component_name) +
                                " DRAM path must be empty (anonymous mmap) or "
                                "start with /dev/shm, got: " + path);
  }
}

struct EngineResolved {
  std::string engine;
  BaseKVConfig cfg;
};

inline EngineResolved ResolveEngine(BaseKVConfig cfg) {
  auto& j = cfg.json_config_;

  const bool has_external_engine_type = j.contains("external_engine_type");
  const bool has_legacy_engine_type   = j.contains("engine_type");
  if (has_external_engine_type || has_legacy_engine_type) {
    const std::string engine =
        has_external_engine_type
            ? j.at("external_engine_type").get<std::string>()
            : j.at("engine_type").get<std::string>();
    if (has_external_engine_type && has_legacy_engine_type) {
      const std::string legacy_engine = j.at("engine_type").get<std::string>();
      if (legacy_engine != engine) {
        throw std::invalid_argument(
            "external_engine_type and deprecated engine_type disagree: " +
            engine + " != " + legacy_engine);
      }
    }
    static const std::set<std::string> kExplicitEngines = {
        "KVEngineFasterKV", "KVEngineHPSHashMap", "KVEngineHPSRocksDB"};
    if (!kExplicitEngines.count(engine)) {
      throw std::invalid_argument("unknown external_engine_type: " + engine);
    }
    for (const char* k : {"path", "capacity"}) {
      if (!j.contains(k)) {
        throw std::invalid_argument(
            std::string(k) + " is required for " + engine);
      }
    }
    const bool has_value_size =
        j.contains("value_size") ||
        (j.contains("value") &&
         j.at("value").contains("default_value_size_hint"));
    if (!has_value_size) {
      throw std::invalid_argument("value_size is required for " + engine);
    }
    return EngineResolved{engine, std::move(cfg)};
  }

  for (const char* k : {"capacity", "index", "value"}) {
    if (!j.contains(k)) {
      throw std::invalid_argument(std::string(k) + " is required");
    }
  }

  for (const char* k :
       {"path",
        "index_type",
        "value_type",
        "allocator_type",
        "value_memory_management",
        "io_backend_type",
        "value_size",
        "engine_type",
        "mode",
        "value_layout",
        "shmcapacity",
        "ssdcapacity"}) {
    if (j.contains(k)) {
      throw std::invalid_argument("legacy field '" + std::string(k) +
                                  "' not allowed; use nested index/value config");
    }
  }

  const auto hasLegacyFilePath = [](const json& node) {
    std::function<bool(const json&)> walk = [&](const json& n) {
      if (n.is_object()) {
        if (n.contains("file_path")) {
          return true;
        }
        for (auto it = n.begin(); it != n.end(); ++it) {
          if (walk(it.value())) {
            return true;
          }
        }
      } else if (n.is_array()) {
        for (const auto& v : n) {
          if (walk(v)) {
            return true;
          }
        }
      }
      return false;
    };
    return walk(node);
  };

  if (hasLegacyFilePath(j)) {
    throw std::invalid_argument(
        "legacy field 'file_path' not allowed; use index.path/value.path/allocator.path");
  }

  const auto& idx_j = j.at("index");
  const auto& val_j = j.at("value");
  if (!idx_j.contains("type")) {
    throw std::invalid_argument("index.type is required");
  }
  if (!val_j.contains("type")) {
    throw std::invalid_argument("value.type is required");
  }

  const std::string idx_type = idx_j.at("type").get<std::string>();
  const std::string val_type = val_j.at("type").get<std::string>();

  static const std::set<std::string> kDramIndex = {
      "DRAM_EXTENDIBLE_HASH", "DRAM_UNORDERED_MAP", "DRAM_PET_HASH"};
  static const std::set<std::string> kValTypes = {
      "DRAM_VALUE_STORE", "SSD_VALUE_STORE", "TIERED_VALUE_STORE"};

  if (!kDramIndex.count(idx_type)) {
    throw std::invalid_argument("unknown index.type: " + idx_type);
  }
  if (!kValTypes.count(val_type)) {
    throw std::invalid_argument("unknown value.type: " + val_type);
  }

  if (val_type == "DRAM_VALUE_STORE") {
    if (!val_j.contains("dram_allocator")) {
      throw std::invalid_argument("DRAM_VALUE_STORE requires value.dram_allocator");
    }
    if (val_j.contains("ssd_allocator")) {
      throw std::invalid_argument("DRAM_VALUE_STORE must not have ssd_allocator");
    }
    if (val_j.contains("dram_allocator") &&
        val_j.at("dram_allocator").contains("path")) {
      throw std::invalid_argument("DRAM_VALUE_STORE must not have value.dram_allocator.path");
    }
    ValidateDramPathPolicy(val_j, "path", "DRAM_VALUE_STORE");
  } else if (val_type == "SSD_VALUE_STORE") {
    if (!val_j.contains("path") || val_j.at("path").get<std::string>().empty()) {
      throw std::invalid_argument("SSD_VALUE_STORE requires non-empty value.path");
    }
    if (!val_j.contains("ssd_allocator")) {
      throw std::invalid_argument("SSD_VALUE_STORE requires value.ssd_allocator");
    }
    if (val_j.contains("dram_allocator")) {
      throw std::invalid_argument("SSD_VALUE_STORE must not have dram_allocator");
    }
    if (val_j.contains("ssd_allocator") &&
        val_j.at("ssd_allocator").contains("path")) {
      throw std::invalid_argument("SSD_VALUE_STORE must not have value.ssd_allocator.path");
    }
  } else if (val_type == "TIERED_VALUE_STORE") {
    if (val_j.contains("path")) {
      throw std::invalid_argument("TIERED_VALUE_STORE must not have value.path");
    }
    if (!val_j.contains("dram_allocator") || !val_j.contains("ssd_allocator")) {
      throw std::invalid_argument(
          "TIERED_VALUE_STORE requires dram_allocator and ssd_allocator");
    }
    const auto& dram = val_j.at("dram_allocator");
    const auto& ssd = val_j.at("ssd_allocator");
    ValidateDramPathPolicy(dram, "path", "TIERED_VALUE_STORE");
    if (!ssd.contains("path") || ssd.at("path").get<std::string>().empty()) {
      throw std::invalid_argument(
          "TIERED_VALUE_STORE requires non-empty value.ssd_allocator.path");
    }
  }

  return EngineResolved{"KVEngineComposite", std::move(cfg)};
}

} // namespace base
