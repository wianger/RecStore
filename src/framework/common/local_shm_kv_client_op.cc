#include "framework/op.h"

#ifndef USE_FAKE_KVCLIENT

#  include "framework/common/hierkv_local_runtime.h"
#  include "framework/common/local_shm_op_component.h"

#  include <stdexcept>

namespace recstore {

void KVClientOp::LocalLookupFlat(const base::RecTensor& keys,
                                 base::RecTensor& values) {
  if (ps_backend_name_ != "local_shm" &&
      !IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "local_lookup_flat requires local_shm or hierkv "
        "backend, "
        "but current backend is " +
        ps_backend_name_);
  }
  if (IsHierKVBackendName(ps_backend_name_)) {
    EmbRead(keys, values);
    return;
  }
  LocalShmLookupFlat(ps_client_, ps_backend_name_, keys, values);
}

int KVClientOp::SubmitLocalLookupFlat(const base::RecTensor& keys,
                                      int64_t embedding_dim,
                                      LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "submit_local_lookup_flat is only supported by local_shm backend.");
  }
  return SubmitLocalShmLookupFlat(
      ps_client_, ps_backend_name_, keys, embedding_dim, handle);
}

int KVClientOp::WaitLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "wait_local_lookup_flat is only supported by local_shm backend.");
  }
  return WaitLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

void KVClientOp::ReleaseLocalLookupFlat(LocalShmFlatGetHandle* handle) {
  if (IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "release_local_lookup_flat is only supported by local_shm backend.");
  }
  ReleaseLocalShmLookupFlat(ps_client_, ps_backend_name_, handle);
}

bool KVClientOp::GetLocalLookupFlatPayloadRegion(const void** base,
                                                 std::size_t* bytes) {
  auto* local_client = GetLocalShmClientOrThrow(
      ps_client_, ps_backend_name_, "warmup_local_lookup_flat_cuda_region");
  return local_client->GetSlotPayloadRegion(base, bytes);
}

void KVClientOp::LocalUpdateFlat(const std::string& table_name,
                                 const base::RecTensor& keys,
                                 const base::RecTensor& grads) {
  if (ps_backend_name_ != "local_shm" &&
      !IsHierKVBackendName(ps_backend_name_)) {
    throw std::runtime_error(
        "local_update_flat requires local_shm or hierkv "
        "backend, "
        "but current backend is " +
        ps_backend_name_);
  }
  if (IsHierKVBackendName(ps_backend_name_)) {
    EmbUpdate(table_name, keys, grads);
    return;
  }
  LocalShmUpdateFlat(ps_client_, ps_backend_name_, table_name, keys, grads);
}

} // namespace recstore

#endif // USE_FAKE_KVCLIENT
