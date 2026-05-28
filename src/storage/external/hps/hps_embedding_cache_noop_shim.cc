#include "third_party/HugeCTR/HugeCTR/include/common.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/embedding_cache_base.hpp"

namespace HugeCTR {

class NoopEmbeddingCache final : public EmbeddingCacheBase {
public:
  explicit NoopEmbeddingCache(const InferenceParams& inference_params) {
    cache_config_.num_emb_table_ =
        inference_params.embedding_vecsize_per_table.size();
    cache_config_.cache_size_percentage_ =
        inference_params.cache_size_percentage;
    cache_config_.cache_refresh_percentage_per_iteration =
        inference_params.cache_refresh_percentage_per_iteration;
    cache_config_.num_set_in_refresh_workspace_ = 0;
    cache_config_.default_value_for_each_table =
        inference_params.default_value_for_each_table;
    cache_config_.model_name_              = inference_params.model_name;
    cache_config_.cuda_dev_id_             = inference_params.device_id;
    cache_config_.use_gpu_embedding_cache_ = false;
    cache_config_.embedding_vec_size_ =
        inference_params.embedding_vecsize_per_table;
    cache_config_.num_set_in_cache_.assign(cache_config_.num_emb_table_, 0);
    cache_config_.embedding_table_name_ =
        inference_params.embedding_table_names;
    cache_config_.max_query_len_per_emb_table_ =
        inference_params.maxnum_catfeature_query_per_table_per_sample;
    cache_config_.use_hctr_cache_implementation =
        inference_params.use_hctr_cache_implementation;
  }

  void
  lookup(size_t, float*, const void*, size_t, float, cudaStream_t) override {
    HCTR_OWN_THROW(Error_t::IllegalCall,
                   "GPU embedding cache is disabled for this HPS benchmark.");
  }

  void lookup_from_device(
      size_t, float*, const void*, size_t, float, cudaStream_t) override {
    HCTR_OWN_THROW(Error_t::IllegalCall,
                   "GPU embedding cache is disabled for this HPS benchmark.");
  }

  void insert(size_t, EmbeddingCacheWorkspace&, cudaStream_t) override {}
  void init(const size_t, EmbeddingCacheRefreshspace&, cudaStream_t) override {}
  void init(const size_t, void*, void*, float*, size_t, cudaStream_t) override {
  }
  void dump(size_t, void*, size_t*, size_t, size_t, cudaStream_t) override {}
  void
  refresh(size_t, const void*, const void*, size_t, cudaStream_t) override {}
  void finalize() override {}
  void insert_stream_for_sync(std::vector<cudaStream_t>) override {}

  EmbeddingCacheWorkspace create_workspace() override {
    EmbeddingCacheWorkspace workspace{};
    workspace.use_gpu_embedding_cache_ = false;
    return workspace;
  }

  void destroy_workspace(EmbeddingCacheWorkspace&) override {}

  EmbeddingCacheRefreshspace create_refreshspace() override {
    return EmbeddingCacheRefreshspace{};
  }

  void destroy_refreshspace(EmbeddingCacheRefreshspace&) override {}

  const embedding_cache_config& get_cache_config() override {
    return cache_config_;
  }

  const std::vector<cudaStream_t>& get_refresh_streams() override {
    return empty_streams_;
  }

  const std::vector<cudaStream_t>& get_insert_streams() override {
    return empty_streams_;
  }

  int get_device_id() override { return cache_config_.cuda_dev_id_; }

  bool use_gpu_embedding_cache() override { return false; }

  void set_profiler(int, int, bool) override {}
  void profiler_print() override {}

private:
  embedding_cache_config cache_config_{};
  std::vector<cudaStream_t> empty_streams_;
};

EmbeddingCacheBase::~EmbeddingCacheBase() = default;

std::shared_ptr<EmbeddingCacheBase> EmbeddingCacheBase::create(
    const InferenceParams& inference_params,
    const parameter_server_config&,
    HierParameterServerBase* const) {
  if (inference_params.use_gpu_embedding_cache) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "GPU embedding cache is not linked in RecStore HPS native "
                   "tiered benchmark.");
  }
  return std::make_shared<NoopEmbeddingCache>(inference_params);
}

} // namespace HugeCTR
