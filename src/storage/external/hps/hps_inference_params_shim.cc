#include "third_party/HugeCTR/HugeCTR/include/common.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/inference_utils.hpp"

#include <algorithm>
#include <filesystem>
#include <thread>
#include <unordered_set>

namespace HugeCTR {

std::optional<size_t>
parameter_server_config::find_model_id(const std::string& model_name) const {
  const auto it = model_name_id_map_.find(model_name);
  if (it != model_name_id_map_.end()) {
    return it->second;
  }
  return std::nullopt;
}

bool VolatileDatabaseParams::operator==(const VolatileDatabaseParams& p) const {
  return type == p.type && address == p.address && user_name == p.user_name &&
         password == p.password && num_partitions == p.num_partitions &&
         allocation_rate == p.allocation_rate &&
         shared_memory_size == p.shared_memory_size &&
         shared_memory_name == p.shared_memory_name &&
         shared_memory_auto_remove == p.shared_memory_auto_remove &&
         num_node_connections == p.num_node_connections &&
         max_batch_size == p.max_batch_size && enable_tls == p.enable_tls &&
         tls_ca_certificate == p.tls_ca_certificate &&
         tls_client_certificate == p.tls_client_certificate &&
         tls_client_key == p.tls_client_key &&
         tls_server_name_identification == p.tls_server_name_identification &&
         overflow_margin == p.overflow_margin &&
         overflow_policy == p.overflow_policy &&
         overflow_resolution_target == p.overflow_resolution_target &&
         initialize_after_startup == p.initialize_after_startup &&
         initial_cache_rate == p.initial_cache_rate &&
         cache_missed_embeddings == p.cache_missed_embeddings &&
         update_filters == p.update_filters;
}

bool VolatileDatabaseParams::operator!=(const VolatileDatabaseParams& p) const {
  return !operator==(p);
}

bool PersistentDatabaseParams::operator==(
    const PersistentDatabaseParams& p) const {
  return type == p.type && path == p.path && num_threads == p.num_threads &&
         read_only == p.read_only && max_batch_size == p.max_batch_size &&
         initialize_after_startup == p.initialize_after_startup &&
         update_filters == p.update_filters;
}

bool PersistentDatabaseParams::operator!=(
    const PersistentDatabaseParams& p) const {
  return !operator==(p);
}

bool UpdateSourceParams::operator==(const UpdateSourceParams& p) const {
  return type == p.type && brokers == p.brokers &&
         metadata_refresh_interval_ms == p.metadata_refresh_interval_ms &&
         receive_buffer_size == p.receive_buffer_size &&
         poll_timeout_ms == p.poll_timeout_ms &&
         max_batch_size == p.max_batch_size &&
         failure_backoff_ms == p.failure_backoff_ms &&
         max_commit_interval == p.max_commit_interval;
}

bool UpdateSourceParams::operator!=(const UpdateSourceParams& p) const {
  return !operator==(p);
}

VolatileDatabaseParams::VolatileDatabaseParams() {
  num_partitions = std::min(
      num_partitions, static_cast<size_t>(std::thread::hardware_concurrency()));
}

VolatileDatabaseParams::VolatileDatabaseParams(
    const DatabaseType_t type,
    const std::string& address,
    const std::string& user_name,
    const std::string& password,
    const size_t num_partitions,
    const size_t allocation_rate,
    const size_t shared_memory_size,
    const std::string& shared_memory_name,
    const bool shared_memory_auto_remove,
    const size_t num_node_connections,
    const size_t max_batch_size,
    const bool enable_tls,
    const std::string& tls_ca_certificate,
    const std::string& tls_client_certificate,
    const std::string& tls_client_key,
    const std::string& tls_server_name_identification,
    const size_t overflow_margin,
    const DatabaseOverflowPolicy_t overflow_policy,
    const double overflow_resolution_target,
    const bool initialize_after_startup,
    const double initial_cache_rate,
    const bool cache_missed_embeddings,
    const std::vector<std::string>& update_filters)
    : type{type},
      address{address},
      user_name{user_name},
      password{password},
      num_partitions{num_partitions},
      allocation_rate{allocation_rate},
      shared_memory_size{shared_memory_size},
      shared_memory_name{shared_memory_name},
      shared_memory_auto_remove{shared_memory_auto_remove},
      num_node_connections{num_node_connections},
      max_batch_size{max_batch_size},
      enable_tls{enable_tls},
      tls_ca_certificate{tls_ca_certificate},
      tls_client_certificate{tls_client_certificate},
      tls_client_key{tls_client_key},
      tls_server_name_identification{tls_server_name_identification},
      overflow_margin{overflow_margin},
      overflow_policy{overflow_policy},
      overflow_resolution_target{overflow_resolution_target},
      initialize_after_startup{initialize_after_startup},
      initial_cache_rate{initial_cache_rate},
      cache_missed_embeddings{cache_missed_embeddings},
      update_filters{update_filters} {}

PersistentDatabaseParams::PersistentDatabaseParams()
    : path{std::filesystem::temp_directory_path() / "rocksdb"} {}

PersistentDatabaseParams::PersistentDatabaseParams(
    const DatabaseType_t type,
    const std::string& path,
    const size_t num_threads,
    const bool read_only,
    const size_t max_batch_size,
    const bool initialize_after_startup,
    const std::vector<std::string>& update_filters)
    : type(type),
      path(path),
      num_threads(num_threads),
      read_only(read_only),
      max_batch_size(max_batch_size),
      initialize_after_startup{initialize_after_startup},
      update_filters(update_filters) {}

UpdateSourceParams::UpdateSourceParams(
    const UpdateSourceType_t type,
    const std::string& brokers,
    const size_t metadata_refresh_interval_ms,
    const size_t receive_buffer_size,
    const size_t poll_timeout_ms,
    const size_t max_batch_size,
    const size_t failure_backoff_ms,
    const size_t max_commit_interval)
    : type(type),
      brokers(brokers),
      metadata_refresh_interval_ms(metadata_refresh_interval_ms),
      receive_buffer_size(receive_buffer_size),
      poll_timeout_ms(poll_timeout_ms),
      max_batch_size(max_batch_size),
      failure_backoff_ms(failure_backoff_ms),
      max_commit_interval(max_commit_interval) {}

InferenceParams::InferenceParams(
    const std::string& model_name,
    const size_t max_batchsize,
    const float hit_rate_threshold,
    const std::string& dense_model_file,
    const std::vector<std::string>& sparse_model_files,
    const int device_id,
    const bool use_gpu_embedding_cache,
    const float cache_size_percentage,
    const bool i64_input_key,
    const bool use_mixed_precision,
    const float scaler,
    const bool use_algorithm_search,
    const bool use_cuda_graph,
    const int number_of_worker_buffers_in_pool,
    const int number_of_refresh_buffers_in_pool,
    const int thread_pool_size,
    const float cache_refresh_percentage_per_iteration,
    const std::vector<int>& deployed_devices,
    const std::vector<float>& default_value_for_each_table,
    const VolatileDatabaseParams& volatile_db,
    const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source,
    const int maxnum_des_feature_per_sample,
    const float refresh_delay,
    const float refresh_interval,
    const std::vector<size_t>& maxnum_catfeature_query_per_table_per_sample,
    const std::vector<size_t>& embedding_vecsize_per_table,
    const std::vector<std::string>& embedding_table_names,
    const std::string& network_file,
    const size_t label_dim,
    const size_t slot_num,
    const std::string& non_trainable_params_file,
    bool use_static_table,
    EmbeddingCacheType_t embedding_cache_type,
    bool use_context_stream,
    bool fuse_embedding_table,
    bool use_hctr_cache_implementation,
    bool init_ec,
    bool enable_pagelock,
    bool fp8_quant)
    : model_name(model_name),
      max_batchsize(max_batchsize),
      hit_rate_threshold(hit_rate_threshold),
      dense_model_file(dense_model_file),
      sparse_model_files(sparse_model_files),
      device_id(device_id),
      use_gpu_embedding_cache(use_gpu_embedding_cache),
      cache_size_percentage(cache_size_percentage),
      i64_input_key(i64_input_key),
      use_mixed_precision(use_mixed_precision),
      scaler(scaler),
      use_algorithm_search(use_algorithm_search),
      use_cuda_graph(use_cuda_graph),
      number_of_worker_buffers_in_pool(number_of_worker_buffers_in_pool),
      number_of_refresh_buffers_in_pool(number_of_refresh_buffers_in_pool),
      thread_pool_size(thread_pool_size),
      cache_refresh_percentage_per_iteration(
          cache_refresh_percentage_per_iteration),
      deployed_devices(deployed_devices),
      default_value_for_each_table(default_value_for_each_table),
      volatile_db(volatile_db),
      persistent_db(persistent_db),
      update_source(update_source),
      maxnum_des_feature_per_sample(maxnum_des_feature_per_sample),
      refresh_delay(refresh_delay),
      refresh_interval(refresh_interval),
      maxnum_catfeature_query_per_table_per_sample(
          maxnum_catfeature_query_per_table_per_sample),
      embedding_vecsize_per_table(embedding_vecsize_per_table),
      embedding_table_names(embedding_table_names),
      network_file(network_file),
      label_dim(label_dim),
      slot_num(slot_num),
      non_trainable_params_file(non_trainable_params_file),
      use_static_table(use_static_table),
      embedding_cache_type(embedding_cache_type),
      use_context_stream(use_context_stream),
      fuse_embedding_table(fuse_embedding_table),
      use_hctr_cache_implementation(use_hctr_cache_implementation),
      init_ec(init_ec),
      enable_pagelock(enable_pagelock),
      fp8_quant(fp8_quant) {
  if (this->default_value_for_each_table.size() !=
      this->sparse_model_files.size()) {
    const float default_value =
        this->default_value_for_each_table.empty()
            ? 0.0f
            : this->default_value_for_each_table[0];
    this->default_value_for_each_table.assign(
        this->sparse_model_files.size(), default_value);
  }
}

parameter_server_config::parameter_server_config(
    std::map<std::string, std::vector<std::string>> emb_table_name,
    std::map<std::string, std::vector<size_t>> embedding_vec_size,
    std::map<std::string, std::vector<size_t>>
        max_feature_num_per_sample_per_emb_table,
    const std::vector<InferenceParams>& inference_params_array,
    const VolatileDatabaseParams& volatile_db,
    const PersistentDatabaseParams& persistent_db,
    const UpdateSourceParams& update_source) {
  if (emb_table_name.size() != inference_params_array.size() ||
      embedding_vec_size.size() != inference_params_array.size() ||
      max_feature_num_per_sample_per_emb_table.size() !=
          inference_params_array.size()) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "Wrong input: inconsistent HPS model parameter maps.");
  }

  for (const auto& inference_params : inference_params_array) {
    if (emb_table_name.find(inference_params.model_name) ==
            emb_table_name.end() ||
        embedding_vec_size.find(inference_params.model_name) ==
            embedding_vec_size.end() ||
        max_feature_num_per_sample_per_emb_table.find(
            inference_params.model_name) ==
            max_feature_num_per_sample_per_emb_table.end()) {
      HCTR_OWN_THROW(Error_t::WrongInput,
                     "Wrong input: model_name does not exist in HPS maps.");
    }

    model_name_id_map_.emplace(
        inference_params.model_name, model_name_id_map_.size());
    emb_file_name_[inference_params.model_name] =
        inference_params.sparse_model_files;
    emb_table_name_[inference_params.model_name] =
        emb_table_name[inference_params.model_name];
    embedding_vec_size_[inference_params.model_name] =
        embedding_vec_size[inference_params.model_name];
    embedding_key_count_[inference_params.model_name] = {};
    max_feature_num_per_sample_per_emb_table_[inference_params.model_name] =
        max_feature_num_per_sample_per_emb_table[inference_params.model_name];
    default_emb_vec_value_.emplace_back(
        inference_params.default_value_for_each_table);
  }

  this->inference_params_array = inference_params_array;
  this->volatile_db            = volatile_db;
  this->persistent_db          = persistent_db;
  this->update_source          = update_source;
  for (auto& inference_params : this->inference_params_array) {
    inference_params.fuse_embedding_table = false;
    inference_params.volatile_db          = volatile_db;
    inference_params.persistent_db        = persistent_db;
    inference_params.update_source        = update_source;
  }
}

parameter_server_config::parameter_server_config(
    const std::vector<std::string>&, const std::vector<InferenceParams>&) {
  HCTR_OWN_THROW(Error_t::WrongInput,
                 "JSON model config is not supported by RecStore HPS shim.");
}

parameter_server_config::parameter_server_config(
    const std::string& hps_json_config_file) {
  init(hps_json_config_file);
}

parameter_server_config::parameter_server_config(
    const char* hps_json_config_file) {
  init(std::string(hps_json_config_file));
}

void parameter_server_config::init(const std::string&) {
  HCTR_OWN_THROW(Error_t::WrongInput,
                 "HPS JSON config is not supported by RecStore HPS shim.");
}

void parameter_server_config::fuse_embedding_table_in_json_config(
    nlohmann::json&) {
  HCTR_OWN_THROW(
      Error_t::WrongInput,
      "HPS JSON table fusion is not supported by RecStore HPS shim.");
}

namespace {

std::string json_string_or_default(const nlohmann::json& json,
                                   const std::string& key,
                                   const std::string& default_value) {
  const auto it = json.find(key);
  if (it == json.end()) {
    return default_value;
  }
  return it->get<std::string>();
}

} // namespace

DatabaseType_t get_hps_database_type(const nlohmann::json& json,
                                     const std::string& key,
                                     const DatabaseType_t default_value) {
  const auto tmp = json_string_or_default(json, key, "");
  if (tmp.empty()) {
    return default_value;
  }
  if (tmp == "disabled" || tmp == "disable" || tmp == "none") {
    return DatabaseType_t::Disabled;
  }
  if (tmp == "hash_map" || tmp == "hashmap" || tmp == "hash" || tmp == "map") {
    return DatabaseType_t::HashMap;
  }
  if (tmp == "parallel_hash_map" || tmp == "parallel_hashmap" ||
      tmp == "parallel_hash" || tmp == "parallel_map") {
    return DatabaseType_t::ParallelHashMap;
  }
  if (tmp == "multi_process_hash_map" || tmp == "multi_process_hashmap" ||
      tmp == "multi_process_hash" || tmp == "multi_process_map") {
    return DatabaseType_t::MultiProcessHashMap;
  }
  if (tmp == "redis_cluster" || tmp == "redis") {
    return DatabaseType_t::RedisCluster;
  }
  if (tmp == "rocks_db" || tmp == "rocksdb" || tmp == "rocks") {
    return DatabaseType_t::RocksDB;
  }
  return default_value;
}

UpdateSourceType_t get_hps_updatesource_type(
    const nlohmann::json& json,
    const std::string& key,
    const UpdateSourceType_t default_value) {
  const auto tmp = json_string_or_default(json, key, "");
  if (tmp == "null" || tmp == "none") {
    return UpdateSourceType_t::Null;
  }
  if (tmp == "kafka_message_queue" || tmp == "kafka_mq" || tmp == "kafka") {
    return UpdateSourceType_t::KafkaMessageQueue;
  }
  return default_value;
}

EmbeddingCacheType_t get_hps_embeddingcache_type(
    const nlohmann::json& json,
    const std::string& key,
    const EmbeddingCacheType_t default_value) {
  const auto tmp = json_string_or_default(json, key, "");
  if (tmp == "dynamic") {
    return EmbeddingCacheType_t::Dynamic;
  }
  if (tmp == "static") {
    return EmbeddingCacheType_t::Static;
  }
  if (tmp == "uvm") {
    return EmbeddingCacheType_t::UVM;
  }
  if (tmp == "stochastic") {
    return EmbeddingCacheType_t::Stochastic;
  }
  return default_value;
}

DatabaseOverflowPolicy_t get_hps_overflow_policy(
    const nlohmann::json& json,
    const std::string& key,
    const DatabaseOverflowPolicy_t default_value) {
  const auto tmp = json_string_or_default(json, key, "");
  if (tmp == "evict_random" || tmp == "random") {
    return DatabaseOverflowPolicy_t::EvictRandom;
  }
  if (tmp == "evict_least_used" || tmp == "least_used") {
    return DatabaseOverflowPolicy_t::EvictLeastUsed;
  }
  if (tmp == "evict_oldest" || tmp == "oldest") {
    return DatabaseOverflowPolicy_t::EvictOldest;
  }
  return default_value;
}

} // namespace HugeCTR
