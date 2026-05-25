#include "storage/io_backend/force_link.h"
#include "storage/kv_engine/engine_composite.h"
#include "storage/kv_engine/engine_petkv.h"

#include "gflags/gflags.h"

namespace {
struct IOBackendLinkGuard {
  IOBackendLinkGuard() { ForceLinkIOBackends(); }
};
const IOBackendLinkGuard kIoBackendLinkGuard;
} // namespace

DEFINE_int32(prefetch_method,
             0,
             "PetKV BatchGet prefetch method: 0=single Get loop, 1=BatchGet");
