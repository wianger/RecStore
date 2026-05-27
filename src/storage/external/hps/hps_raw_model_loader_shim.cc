#include "third_party/HugeCTR/HugeCTR/include/common.hpp"
#include "third_party/HugeCTR/HugeCTR/include/hps/modelloader.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <type_traits>

namespace HugeCTR {

namespace {

template <typename T>
void read_exact(const std::string& path,
                std::vector<T>& out,
                size_t elements,
                size_t element_offset) {
  out.resize(elements);
  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    HCTR_OWN_THROW(Error_t::WrongInput, "failed to open HPS raw model file");
  }
  in.seekg(static_cast<std::streamoff>(element_offset * sizeof(T)));
  in.read(reinterpret_cast<char*>(out.data()),
          static_cast<std::streamsize>(elements * sizeof(T)));
  if (!in) {
    HCTR_OWN_THROW(Error_t::WrongInput, "failed to read HPS raw model file");
  }
}

template <typename TKey>
void read_keys(const std::string& path,
               std::vector<TKey>& out,
               size_t elements,
               size_t element_offset) {
  if constexpr (std::is_same<TKey, long long>::value) {
    read_exact(path, out, elements, element_offset);
  } else {
    std::vector<long long> i64_keys;
    read_exact(path, i64_keys, elements, element_offset);
    out.resize(elements);
    std::transform(
        i64_keys.begin(), i64_keys.end(), out.begin(), [](long long key) {
          return static_cast<unsigned int>(key);
        });
  }
}

} // namespace

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_cache_keys() {
  return this->keys.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_caceh_vecs() {
  return this->vectors.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_uvm_keys() {
  return this->uvm_keys.data();
}

template <typename TKey, typename TValue>
void* UnifiedEmbeddingTable<TKey, TValue>::get_uvm_vecs() {
  return this->uvm_vectors.data();
}

template <typename TKey, typename TValue>
size_t UnifiedEmbeddingTable<TKey, TValue>::get_cache_key_count() {
  return this->key_count;
}

template <typename TKey, typename TValue>
size_t UnifiedEmbeddingTable<TKey, TValue>::get_uvm_key_count() {
  return this->uvm_key_count;
}

template <typename TKey, typename TValue>
RawModelLoader<TKey, TValue>::RawModelLoader() : IModelLoader() {
  embedding_table_ = new UnifiedEmbeddingTable<TKey, TValue>();
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load_fused_emb(
    const std::string& table_name, const std::vector<std::string>& path_list) {
  delete_table();
  embedding_table_ = new UnifiedEmbeddingTable<TKey, TValue>();
  for (const auto& path : path_list) {
    load_emb(table_name, path);
  }
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load_emb(const std::string&,
                                            const std::string& path) {
  const std::string key_file = path + "/key";
  const std::string vec_file = path + "/emb_vector";
  const auto key_file_size   = std::filesystem::file_size(key_file);
  const auto vec_file_size   = std::filesystem::file_size(vec_file);
  if (key_file_size == 0 || vec_file_size == 0 ||
      key_file_size % sizeof(long long) != 0 ||
      vec_file_size % sizeof(TValue) != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "invalid HPS raw model file size");
  }
  const size_t key_count = key_file_size / sizeof(long long);
  const size_t vec_count = vec_file_size / sizeof(TValue);

  const size_t key_offset = embedding_table_->key_count;
  const size_t vec_offset = embedding_table_->vec_elem_count;
  embedding_table_->key_count += key_count;
  embedding_table_->total_key_count = embedding_table_->key_count;
  embedding_table_->vec_elem_count += vec_count;
  embedding_table_->keys.resize(embedding_table_->key_count);
  embedding_table_->vectors.resize(embedding_table_->vec_elem_count);

  std::vector<TKey> keys;
  read_keys(key_file, keys, key_count, 0);
  std::copy(
      keys.begin(), keys.end(), embedding_table_->keys.begin() + key_offset);

  std::vector<TValue> values;
  read_exact(vec_file, values, vec_count, 0);
  std::copy(values.begin(),
            values.end(),
            embedding_table_->vectors.begin() + vec_offset);
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::load(
    const std::string&,
    const std::string& path,
    size_t key_num_per_iteration,
    size_t,
    bool fp8_quant) {
  if (fp8_quant) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "fp8 raw model loading is not supported by RecStore shim");
  }
  embedding_folder_path      = path;
  const std::string key_file = path + "/key";
  const std::string vec_file = path + "/emb_vector";
  const auto key_file_size   = std::filesystem::file_size(key_file);
  const auto vec_file_size   = std::filesystem::file_size(vec_file);
  if (key_file_size == 0 || vec_file_size == 0 ||
      key_file_size % sizeof(long long) != 0 ||
      vec_file_size % sizeof(TValue) != 0) {
    HCTR_OWN_THROW(Error_t::WrongInput, "invalid HPS raw model file size");
  }
  embedding_table_->total_key_count = key_file_size / sizeof(long long);
  key_iteration =
      key_num_per_iteration == 0
          ? std::max<size_t>(1, embedding_table_->total_key_count / 10)
          : key_num_per_iteration;
  num_iterations =
      (embedding_table_->total_key_count + key_iteration - 1) / key_iteration;
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::delete_table() {
  if (!embedding_table_) {
    return;
  }
  delete embedding_table_;
  embedding_table_ = nullptr;
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getkeys() {
  return embedding_table_->keys.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getvectors() {
  return embedding_table_->vectors.data();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::getmetas(bool) {
  return embedding_table_->meta.data();
}

template <typename TKey, typename TValue>
void RawModelLoader<TKey, TValue>::get_cache_uvm(size_t, size_t, size_t) {
  HCTR_OWN_THROW(Error_t::WrongInput,
                 "cache/uvm raw loading is not supported by RecStore shim");
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::getkeycount() {
  return embedding_table_->total_key_count;
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_num_iterations() {
  return num_iterations;
}

template <typename TKey, typename TValue>
std::pair<void*, size_t>
RawModelLoader<TKey, TValue>::getkeys(size_t iteration) {
  const size_t start = iteration * key_iteration;
  const size_t count =
      std::min(key_iteration, embedding_table_->total_key_count - start);
  read_keys(
      embedding_folder_path + "/key", embedding_table_->keys, count, start);
  return {embedding_table_->keys.data(), count};
}

template <typename TKey, typename TValue>
std::pair<void*, size_t> RawModelLoader<TKey, TValue>::getvectors(
    size_t iteration, size_t emb_size, bool fp8_quant) {
  if (fp8_quant) {
    HCTR_OWN_THROW(Error_t::WrongInput,
                   "fp8 raw model loading is not supported by RecStore shim");
  }
  const size_t start_key = iteration * key_iteration;
  const size_t key_count =
      std::min(key_iteration, embedding_table_->total_key_count - start_key);
  const size_t value_count = key_count * emb_size;
  read_exact(embedding_folder_path + "/emb_vector",
             embedding_table_->vectors,
             value_count,
             start_key * emb_size);
  return {embedding_table_->vectors.data(), value_count};
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_cache_keys() {
  return embedding_table_->get_cache_keys();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_caceh_vecs() {
  return embedding_table_->get_caceh_vecs();
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_cache_key_count() {
  return embedding_table_->get_cache_key_count();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_uvm_keys() {
  return embedding_table_->get_uvm_keys();
}

template <typename TKey, typename TValue>
void* RawModelLoader<TKey, TValue>::get_uvm_vecs() {
  return embedding_table_->get_uvm_vecs();
}

template <typename TKey, typename TValue>
size_t RawModelLoader<TKey, TValue>::get_uvm_key_count() {
  return embedding_table_->get_uvm_key_count();
}

template class RawModelLoader<long long, float>;
template class RawModelLoader<unsigned int, float>;

} // namespace HugeCTR
