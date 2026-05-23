#include <folly/init/Init.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "base/base.h"
#include "base/bind_core.h"
#include "base/factory.h"
#include "base/init.h"
#include "base/log.h"
#include "base/string.h"
#include "base/timer.h"
#include "mayfly_config.h"
#include "memory/epoch_manager.h"
#include "memory/shm_file.h"
#include "petps_magic.h"
#include "ps/base/Postoffice.h"
#include "ps/base/base_ps_server.h"
#include "ps/base/cache_ps_impl.h"
#include "ps/base/parameters.h"
#include "ps/base/shard_manager.h"
#include "ps/rdma/rdma_protocol.h"
#include "ps/rdma/rdma_status.h"
#include "src/base/config.h"
#include "third_party/Mayfly-main/include/DSM.h"
#include "third_party/Mayfly-main/include/RawMessageConnection.h"
#include "third_party/json/single_include/nlohmann/json.hpp"

DEFINE_string(config_path, "", "config file path");

DEFINE_double(warmup_ratio,
              0.8,
              "bulk load (warmup_ratio * key_space) kvs in DB");

DEFINE_int32(thread_num, 1, "");
DEFINE_bool(use_sglist, true, "");
DEFINE_bool(preload, false, "");
DEFINE_bool(use_dram, false, "");
DEFINE_int32(numa_id, 0, "");
DEFINE_uint64(rdma_per_thread_response_limit_bytes,
              1 * 1024 * 1024,
              "Per-thread max response bytes for RDMA GET replies");

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

namespace {
constexpr std::size_t kRdmaThreadBufferBytes = 1 * define::MB;

bool ShouldValidateRouting() {
  const char* env = std::getenv("RECSTORE_RDMA_VALIDATE_ROUTING");
  return env != nullptr && std::string(env) != "0";
}

bool ShouldTraceRdmaGet() {
  static const bool enabled = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE");
    return env != nullptr && std::string(env) != "0";
  }();
  return enabled;
}

std::uint64_t RdmaGetTraceInterval() {
  static const std::uint64_t interval = [] {
    const char* env = std::getenv("RECSTORE_RDMA_GET_TRACE_INTERVAL");
    if (env == nullptr) {
      return std::uint64_t{5000};
    }
    const auto parsed =
        static_cast<std::uint64_t>(std::strtoull(env, nullptr, 10));
    return parsed == 0 ? std::uint64_t{5000} : parsed;
  }();
  return interval;
}

std::uint64_t ToNs(std::chrono::steady_clock::duration duration) {
  return static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count());
}

struct RdmaGetServerHandleTraceStats {
  std::atomic<std::uint64_t> count{0};
  std::atomic<std::uint64_t> parse_ns{0};
  std::atomic<std::uint64_t> index_ns{0};
  std::atomic<std::uint64_t> response_ns{0};
  std::atomic<std::uint64_t> write_ns{0};
  std::atomic<std::uint64_t> total_ns{0};

  void Add(const char* mode,
           std::uint64_t parse,
           std::uint64_t index,
           std::uint64_t response,
           std::uint64_t write,
           std::uint64_t total) {
    parse_ns.fetch_add(parse, std::memory_order_relaxed);
    index_ns.fetch_add(index, std::memory_order_relaxed);
    response_ns.fetch_add(response, std::memory_order_relaxed);
    write_ns.fetch_add(write, std::memory_order_relaxed);
    total_ns.fetch_add(total, std::memory_order_relaxed);
    const auto n        = count.fetch_add(1, std::memory_order_relaxed) + 1;
    const auto interval = RdmaGetTraceInterval();
    if (n % interval != 0) {
      return;
    }
    const double denom = static_cast<double>(n) * 1000.0;
    std::ostringstream out;
    out << "component=rdma_get_trace side=server stage=handle mode=" << mode
        << " count=" << n
        << " parse_us_avg=" << parse_ns.load(std::memory_order_relaxed) / denom
        << " index_us_avg=" << index_ns.load(std::memory_order_relaxed) / denom
        << " response_us_avg="
        << response_ns.load(std::memory_order_relaxed) / denom
        << " write_us_avg=" << write_ns.load(std::memory_order_relaxed) / denom
        << " total_us_avg=" << total_ns.load(std::memory_order_relaxed) / denom;
    LOG(INFO) << out.str();
    std::cerr << out.str() << std::endl;
  }
};

RdmaGetServerHandleTraceStats& RawGetServerHandleTraceStats() {
  static RdmaGetServerHandleTraceStats stats;
  return stats;
}

struct RequestView {
  RpcType type;
  NodeIDType node_id;
  ThreadIDType t_id;
  GlobalAddress receive_gaddr;
  Slice payload;
};

RequestView RequestViewFromRawMessage(RawMessage* recv) {
  Cursor cursor;
  return RequestView{
      recv->type,
      recv->node_id,
      recv->t_id,
      recv->receive_gaddr,
      recv->get_string(cursor),
  };
}
} // namespace

namespace recstore {

class PetPSServer : public BaseParameterServer {
public:
  PetPSServer(CachePS* cache_ps, int thread_count)
      : cache_ps_(cache_ps),
        thread_count_(thread_count),
        get_parameter_timer_("GetParameter", 1),
        index_timer_("Index Part", 1),
        value_timer_("Value Part", 1),
        epoch_manager_(base::epoch::EpochManager::GetInstance()) {
    CHECK_LE(thread_count, kMaxThread);

    ClusterInfo cluster;
    cluster.serverNR = XPostoffice::GetInstance()->NumServers();
    cluster.clientNR = XPostoffice::GetInstance()->NumClients();

    DSMConfig config(CacheConfig(), cluster, 0, false);
    if (FLAGS_use_sglist) {
      LOG(WARNING) << "PM address registration not implemented for cache_ps, "
                      "using default DRAM allocation";
      config.dsmSize  = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "Using DRAM space instead of PM space";
    } else {
      config.dsmSize  = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "WE DONT register PM space to RNIC";
    }
    LOG(INFO) << "register MR start =" << (void*)config.baseAddr
              << ", end = " << (void*)(config.baseAddr + config.dsmSize)
              << ", size = " << config.dsmSize;

    config.NIC_name = '0' + FLAGS_numa_id;
    dsm_ = DSM::getInstance(config, XPostoffice::GetInstance()->GlobalID());
    CHECK_EQ(dsm_->getMyNodeID(), XPostoffice::GetInstance()->GlobalID())
        << "inconsistent postoffice and wq dsm";
    LOG(INFO) << "xmh: finish construct DSM";

    sourcelists_.resize(thread_count);
    for (int i = 0; i < thread_count; i++) {
      sourcelists_[i].resize(FLAGS_max_kv_num_per_request);
    }
  }

  void Run() {
    for (int i = 0; i < thread_count_; i++) {
      LOG(INFO) << "Starts PS polling thread " << i;
      threads_.emplace_back(&PetPSServer::PollingThread, this, i);
      tp[i][0] = 0;
    }
  }

  uint64_t GetThroughputCounterSum() const {
    uint64_t sum = 0;
    for (int i = 0; i < thread_count_; i++) {
      sum += tp[i][0];
    }
    return sum;
  }

private:
  void PublishReadyKeys() {
    if (ready_published_.exchange(true, std::memory_order_acq_rel)) {
      return;
    }
    const std::string ready_key =
        "petps-server-ready-" +
        std::to_string(XPostoffice::GetInstance()->ServerID());
    XPostoffice::GetInstance()->MemCachedSet(ready_key, "1");
    VLOG(1) << "component=rdma_server event=publish_ready_key key="
            << ready_key;
  }

  void RpcGetServerServingThreadIDs(RawMessage* recv) {
    CHECK_EQ(recv->type, GET_SERVER_THREADIDS);
    VLOG(1) << "component=rdma_server event=get_server_threadids_recv node_id="
            << static_cast<int>(recv->node_id)
            << " tid=" << static_cast<int>(recv->t_id);
    static std::atomic_int serving_thread_id{0};
    auto m  = RawMessage::get_new_msg();
    m->type = RESP_GET_SERVER_THREADIDS;
    std::vector<int> thread_ids;
    thread_ids.reserve(static_cast<std::size_t>(thread_count_));
    const int start = serving_thread_id.fetch_add(1);
    for (int i = 0; i < thread_count_; ++i) {
      thread_ids.push_back((start + i) % thread_count_);
    }
    dsm_->rpc_call(
        m,
        recv->node_id,
        recv->t_id,
        Slice((char*)thread_ids.data(), thread_ids.size() * sizeof(int)));
  }

  void HandlePsPut(const RequestView& request, int thread_id) {
    std::string error;
    petps::DecodedPutPayload decoded;
    petps::RpcStatus status = petps::RpcStatus::kOk;
    const std::string_view payload_view(request.payload.s, request.payload.len);
    if (!petps::DecodePutPayload(payload_view, &decoded, &error)) {
      LOG(ERROR) << "RpcPsPut decode error: " << error;
      status = petps::RpcStatus::kInvalidPayload;
    } else if (decoded.embedding_dim * sizeof(float) != FLAGS_value_size) {
      LOG(ERROR) << "RpcPsPut value size mismatch, embedding_dim="
                 << decoded.embedding_dim
                 << " FLAGS_value_size=" << FLAGS_value_size;
      status = petps::RpcStatus::kValueSizeMismatch;
    } else if (decoded.keys.size() >
               static_cast<std::size_t>(FLAGS_max_kv_num_per_request)) {
      LOG(ERROR) << "RpcPsPut batch too large, key_count="
                 << decoded.keys.size()
                 << " max_kv_num_per_request=" << FLAGS_max_kv_num_per_request;
      status = petps::RpcStatus::kBatchTooLarge;
    } else {
      cache_ps_->PutDenseParameterBatch(
          decoded.keys.data(),
          decoded.values.data(),
          static_cast<int>(decoded.keys.size()),
          static_cast<int>(decoded.embedding_dim),
          thread_id);
    }

    const std::int32_t code = static_cast<std::int32_t>(status);
    auto* ack_buf           = dsm_->get_rdma_buffer();
    std::memcpy(ack_buf, &code, sizeof(code));
    dsm_->write(
        ack_buf, request.receive_gaddr, sizeof(code), true, petps::WR_ID_PUT);
  }

  void RpcPsPut(RawMessage* recv, int thread_id) {
    HandlePsPut(RequestViewFromRawMessage(recv), thread_id);
  }

  void RpcPsGet(RawMessage* recv, int thread_id) {
    HandlePsGet(RequestViewFromRawMessage(recv), thread_id);
  }

  void HandlePsGet(const RequestView& request, int thread_id) {
    const bool trace_get = ShouldTraceRdmaGet();
    const auto trace_start =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    const bool perf_condition = (thread_id == 0);
    auto& sourcelist          = sourcelists_[thread_id];
    (void)sourcelist;

    epoch_manager_->Protect();

    if (perf_condition) {
      get_parameter_timer_.start();
    }
    Slice extra_data = request.payload;

    const int batch_get_kv_count = extra_data.len / sizeof(uint64_t);
    tp[thread_id][0] += batch_get_kv_count;
    base::ConstArray<uint64_t> keys(
        (uint64_t*)extra_data.s, batch_get_kv_count);
    if (ShouldValidateRouting()) {
      for (auto each : keys) {
        CHECK_EQ(XPostoffice::GetInstance()->ServerID(),
                 ShardManager::KeyPartition(each))
            << each << " not belong to this PS; "
            << "sended from client node_id = " << (int)request.node_id;
      }
    }
    CHECK_LE(batch_get_kv_count, FLAGS_max_kv_num_per_request);
    const auto trace_after_parse =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

    const int embedding_dim = FLAGS_value_size / sizeof(float);
    const std::size_t response_bytes =
        petps::FixedSlotResponseBytes(batch_get_kv_count, FLAGS_value_size);
    auto* buf = dsm_->get_rdma_buffer();

    if (perf_condition) {
      index_timer_.start();
    }
    bool flat_get_ok = true;
    if (response_bytes <= FLAGS_rdma_per_thread_response_limit_bytes) {
      flat_get_ok = cache_ps_->GetParameterFlat(
          keys,
          reinterpret_cast<float*>(buf),
          batch_get_kv_count,
          embedding_dim,
          thread_id);
    }
    if (perf_condition) {
      index_timer_.end();
    }
    const auto trace_after_index =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};

    if (perf_condition) {
      value_timer_.start();
    }

    if (response_bytes > FLAGS_rdma_per_thread_response_limit_bytes) {
      LOG(ERROR) << "component=rdma_server event=batch_too_large shard="
                 << XPostoffice::GetInstance()->ServerID() << " thread_id="
                 << thread_id << " key_count=" << batch_get_kv_count
                 << " response_bytes=" << response_bytes << " limit_bytes="
                 << FLAGS_rdma_per_thread_response_limit_bytes;
      auto* status_word =
          reinterpret_cast<std::int32_t*>(dsm_->get_rdma_buffer());
      *status_word =
          static_cast<std::int32_t>(petps::RpcStatus::kBatchTooLarge);
      dsm_->write(reinterpret_cast<const char*>(status_word),
                  request.receive_gaddr,
                  sizeof(std::int32_t),
                  true,
                  petps::WR_ID_GET);
      epoch_manager_->UnProtect();
      return;
    }

    auto* status_word = reinterpret_cast<std::int32_t*>(
        buf + batch_get_kv_count * FLAGS_value_size);
    if (flat_get_ok) {
      *status_word = static_cast<std::int32_t>(petps::RpcStatus::kOk);
    } else {
      *status_word =
          static_cast<std::int32_t>(petps::RpcStatus::kValueSizeMismatch);
    }

    epoch_manager_->UnProtect();
    const auto trace_before_write =
        trace_get ? std::chrono::steady_clock::now()
                  : std::chrono::steady_clock::time_point{};
    dsm_->write(
        buf, request.receive_gaddr, response_bytes, true, petps::WR_ID_GET);
    if (trace_get) {
      const auto trace_done = std::chrono::steady_clock::now();
      RawGetServerHandleTraceStats().Add(
          "raw_message",
          ToNs(trace_after_parse - trace_start),
          ToNs(trace_after_index - trace_after_parse),
          ToNs(trace_before_write - trace_after_index),
          ToNs(trace_done - trace_before_write),
          ToNs(trace_done - trace_start));
    }

    if (perf_condition) {
      value_timer_.end();
      get_parameter_timer_.end();
    }
  }

  void PollingThread(int thread_id) {
    base::auto_bind_core();
    dsm_->registerThread();
    VLOG(1) << "component=rdma_server event=polling_thread_ready thread_id="
            << thread_id;
    const int ready_threads = registered_polling_threads_.fetch_add(1) + 1;
    if (ready_threads == thread_count_) {
      PublishReadyKeys();
    }

    auto msg = RawMessage::get_new_msg();
    while (1) {
      msg->clear();
      uint64_t wr_id;
      RawMessage* recv;
      do {
        recv = dsm_->rpc_fast_wait(&wr_id);
        if (recv == nullptr && wr_id == petps::WR_ID_SG_GET) {
          epoch_manager_->UnProtect();
        }
      } while (nullptr == recv);

      if (recv->type == GET_SERVER_THREADIDS) {
        RpcGetServerServingThreadIDs(recv);
      } else if (recv->type == PUT) {
        RpcPsPut(recv, thread_id);
      } else if (recv->type == GET) {
        RpcPsGet(recv, thread_id);
      } else {
        LOG(FATAL) << "unknown message type";
      }
    }
  }

private:
  std::vector<std::vector<SourceList>> sourcelists_;
  CachePS* cache_ps_;
  std::vector<std::thread> threads_;
  int thread_count_;
  DSM* dsm_;
  std::atomic_int registered_polling_threads_{0};
  std::atomic<bool> ready_published_{false};
  xmh::Timer get_parameter_timer_;
  xmh::Timer index_timer_;
  xmh::Timer value_timer_;
  base::epoch::EpochManager* epoch_manager_;

  constexpr static int kMaxThread = 128;
  uint64_t tp[kMaxThread][8];
};
} // namespace recstore

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv);
  if (ShouldTraceRdmaGet()) {
    std::cerr << "component=rdma_get_trace side=server event=enabled interval="
              << RdmaGetTraceInterval() << std::endl;
  }
  xmh::Reporter::StartReportThread();

  base::PMMmapRegisterCenter::GetConfig().backend =
      base::PMMmapRegisterCenter::BackendFromUseDram(FLAGS_use_dram);
  base::PMMmapRegisterCenter::GetConfig().numa_id  = FLAGS_numa_id;

  extern int global_socket_id;
  global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  const std::string config_path =
      FLAGS_config_path.empty()
          ? base::ResolveRecStoreConfigPath().string()
          : FLAGS_config_path;
  std::ifstream config_file(config_path);
  if (!config_file.is_open()) {
    LOG(FATAL) << "Cannot open config file: " << config_path;
  }
  nlohmann::json config;
  config_file >> config;

  auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);
  if (cache_ps == nullptr) {
    LOG(FATAL) << "Cannot construct cache_ps";
  }
  auto ps =
      std::make_unique<recstore::PetPSServer>(cache_ps.get(), FLAGS_thread_num);
  ps->Run();
  while (1) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  return 0;
}
