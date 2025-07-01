#include <folly/init/Init.h>

#include <cstdint>
#include <fstream>
#include <future>
#include <string>
#include <vector>

#include "base/base.h"
#include "base/factory.h"
#include "base/log.h"
#include "base/timer.h"
#include "base_ps/Postoffice.h"
#include "base_ps/base_ps_server.h"
#include "base_ps/cache_ps_impl.h"
#include "base_ps/parameters.h"
#include "mayfly_config.h"
#include "memory/epoch_manager.h"
#include "memory/shm_file.h"
#include "petps_magic.h"
#include "recstore_config.h"
#include "third_party/Mayfly-main/include/DSM.h"
#include "third_party/json/single_include/nlohmann/json.hpp"

DEFINE_string(config_path, RECSTORE_PATH "/recstore_config.json",
              "config file path");

DEFINE_double(warmup_ratio, 0.8,
              "bulk load (warmup_ratio * key_space) kvs in DB");

DEFINE_int32(thread_num, 1, "");
DEFINE_bool(use_sglist, true, "");
DEFINE_bool(preload, false, "");
DEFINE_bool(use_dram, false, "");
DEFINE_int32(numa_id, 0, "");

DECLARE_int32(value_size);
DECLARE_int32(max_kv_num_per_request);

namespace recstore {
class PetPSServer : public BaseParameterServer {
 public:
  PetPSServer(CachePS *cache_ps, int thread_count)
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
      // TODO: Need to implement PM address registration for cache_ps
      LOG(WARNING) << "PM address registration not implemented for cache_ps, "
                      "using default DRAM allocation";
      config.dsmSize = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "Using DRAM space instead of PM space";
    } else {
      config.dsmSize = 100 * define::MB;
      config.baseAddr = (uint64_t)hugePageAlloc(config.dsmSize);
      LOG(INFO) << "WE DONT register PM space to RNIC";
    }
    LOG(INFO) << "register MR start =" << (void *)config.baseAddr
              << ", end = " << (void *)(config.baseAddr + config.dsmSize)
              << ", size = " << config.dsmSize;

    config.NIC_name = '0' + FLAGS_numa_id;
    {
      dsm_ = DSM::getInstance(config, XPostoffice::GetInstance()->GlobalID());
      CHECK_EQ(dsm_->getMyNodeID(), XPostoffice::GetInstance()->GlobalID())
          << "inconsistent postoffice and wq dsm";
      LOG(INFO) << "xmh: finish construct DSM";
    }
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
    for (int i = 0; i < thread_count_; i++) sum += tp[i][0];
    return sum;
  }

 private:
  void RpcGetServerServingThreadIDs(RawMessage *recv) {
    CHECK_EQ(recv->type, GET_SERVER_THREADIDS);
    static std::atomic_int serving_thread_id{0};
    auto m = RawMessage::get_new_msg();
    m->type = RESP_GET_SERVER_THREADIDS;
    std::vector<int> thread_ids;
    int r = serving_thread_id.fetch_add(1);
    thread_ids.push_back(r % thread_count_);
    r = serving_thread_id.fetch_add(1);
    thread_ids.push_back(r % thread_count_);
    dsm_->rpc_call(
        m, recv->node_id, recv->t_id,
        Slice((char *)thread_ids.data(), thread_ids.size() * sizeof(int)));
  }

  void RpcPsPut(RawMessage *recv, int thread_id) {
    thread_local base::PseudoRandom random_engine;
    Cursor cursor;
    Slice extra_data = recv->get_string(cursor);
    int put_kv_count = extra_data.len / sizeof(uint64_t);
    base::ConstArray<uint64_t> keys((uint64_t *)extra_data.s, put_kv_count);

    // warning (now direct insert fake value)
    for (int i = 0; i < keys.Size(); i++) {
      cache_ps_->PutSingleParameter(
          keys[i], random_engine.GetString(FLAGS_value_size).c_str(),
          FLAGS_value_size / sizeof(float), thread_id);
    }

    auto buf = dsm_->get_rdma_buffer();
    memcpy(buf, "123", 4);
    GlobalAddress gaddr = recv->receive_gaddr;
    dsm_->write(buf, gaddr, 4, true, petps::WR_ID_PUT);
  }

  void RpcPsGet(RawMessage *recv, int thread_id) {
    thread_local std::vector<ParameterPack> parameter_packs;
    const bool perf_condition = (thread_id == 0);
    auto &sourcelist = sourcelists_[thread_id];

    epoch_manager_->Protect();

    if (perf_condition) get_parameter_timer_.start();
    Cursor cursor;
    Slice extra_data = recv->get_string(cursor);

    int batch_get_kv_count = extra_data.len / sizeof(uint64_t);
    tp[thread_id][0] += batch_get_kv_count;
    base::ConstArray<uint64_t> keys((uint64_t *)extra_data.s,
                                    batch_get_kv_count);
#ifdef RPC_DEBUG
    for (auto each : keys) {
      CHECK_EQ(XPostoffice::GetInstance()->ServerID(),
               ShardManager::KeyPartition(each))
          << each << " not belong to this PS; "
          << "sended from client node_id = " << (int)recv->node_id;
      ;
    }
    LOG(INFO) << "recv->msg_size=" << extra_data.len;
    LOG(INFO) << "server batch gets: " << keys.Debug();
#endif
    CHECK_LE(batch_get_kv_count, FLAGS_max_kv_num_per_request);
    parameter_packs.clear();
    parameter_packs.resize(batch_get_kv_count);

    if (perf_condition) index_timer_.start();
    // Use cache_ps to get parameters
    for (int i = 0; i < batch_get_kv_count; i++) {
      cache_ps_->GetParameterRun2Completion(keys[i], parameter_packs[i],
                                            thread_id);
    }
    if (perf_condition) index_timer_.end();

#ifdef RPC_DEBUG
    int emb_dim = FLAGS_value_size / sizeof(float);
    for (int i = 0; i < batch_get_kv_count; i++) {
      if (parameter_packs[i].dim > 0) {
        XDebug::AssertTensorEq(
            parameter_packs[i].emb_data, emb_dim, keys[i],
            folly::sformat("server embedding check error, key is {}", keys[i]));
      }
    }
#endif
    if (perf_condition) value_timer_.start();
    if (FLAGS_use_sglist) {
      // Note: Simplified implementation - PM address checking disabled for
      // cache_ps
      for (int i = 0; i < batch_get_kv_count; i++) {
        if (parameter_packs[i].dim > 0) {
          sourcelist[i].addr = (void *)parameter_packs[i].emb_data;
          sourcelist[i].size = parameter_packs[i].dim * sizeof(float);
        } else {
          // Handle missing keys
          sourcelist[i].addr = nullptr;
          sourcelist[i].size = 0;
        }
      }

      GlobalAddress gaddr = recv->receive_gaddr;
      CHECK(dsm_->write_from_pm_vec(sourcelist.data(), batch_get_kv_count,
                                    gaddr, true, 30, petps::WR_ID_SG_GET));
    } else {
      auto buf = dsm_->get_rdma_buffer();
      int acc = 0;
      for (int i = 0; i < batch_get_kv_count; i++) {
        if (parameter_packs[i].dim > 0) {
          memcpy(buf + acc, parameter_packs[i].emb_data,
                 parameter_packs[i].dim * sizeof(float));
          acc += parameter_packs[i].dim * sizeof(float);
        }
      }
      epoch_manager_->UnProtect();
      GlobalAddress gaddr = recv->receive_gaddr;
      dsm_->write(buf, gaddr, acc, true, petps::WR_ID_GET);
    }
    if (perf_condition) value_timer_.end();

#ifdef RPC_DEBUG
    LOG(INFO) << "RPC done";
#endif
    if (perf_condition) get_parameter_timer_.end();
  }

  void PollingThread(int thread_id) {
    auto_bind_core(0);
    dsm_->registerThread();
    auto msg = RawMessage::get_new_msg();

    while (1) {
      msg->clear();
      uint64_t wr_id;
      RawMessage *recv;
      do {
        recv = dsm_->rpc_fast_wait(&wr_id);
        if (recv == nullptr && wr_id == petps::WR_ID_SG_GET) {
          // FB_LOG_EVERY_MS(ERROR, 1000)
          //     << "MaxPendingEpochNumPerThread = "
          //     << epoch_manager_->MaxPendingEpochNumPerThread();
          epoch_manager_->UnProtect();
        }
      } while (nullptr == recv);

      if (recv->type == GET_SERVER_THREADIDS) {
        LOG(INFO) << "RPC: GET_SERVER_THREADIDS received";
        RpcGetServerServingThreadIDs(recv);
      } else if (recv->type == PUT) {
        FB_LOG_EVERY_MS(WARNING, 5000) << "here is write";
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
  CachePS *cache_ps_;
  std::vector<std::thread> threads_;
  int thread_count_;
  DSM *dsm_;
  xmh::Timer get_parameter_timer_;
  xmh::Timer index_timer_;
  xmh::Timer value_timer_;

  base::epoch::EpochManager *epoch_manager_;

  constexpr static int kMaxThread = 128;
  uint64_t tp[kMaxThread][8];
};
}  // namespace recstore

int main(int argc, char *argv[]) {
  folly::init(&argc, &argv);
  xmh::Reporter::StartReportThread();

  base::PMMmapRegisterCenter::GetConfig().use_dram = FLAGS_use_dram;
  base::PMMmapRegisterCenter::GetConfig().numa_id = FLAGS_numa_id;

  extern int global_socket_id;
  global_socket_id = FLAGS_numa_id;
  LOG(INFO) << "set NUMA ID = " << FLAGS_numa_id;

  std::ifstream config_file(FLAGS_config_path);
  if (!config_file.is_open()) {
    LOG(FATAL) << "Cannot open config file: " << FLAGS_config_path;
  }
  nlohmann::json config;
  config_file >> config;

  auto cache_ps = std::make_unique<CachePS>(config["cache_ps"]);

  if (FLAGS_preload) {
    LOG(INFO) << "Loading fake data for preload";
    int64_t ps_capacity = config["cache_ps"]["base_kv_config"]["capacity"];
    cache_ps->LoadFakeData(ps_capacity * FLAGS_warmup_ratio, FLAGS_value_size);
  }

  recstore::PetPSServer parameterServiceImpl(cache_ps.get(), FLAGS_thread_num);
  parameterServiceImpl.Run();

  while (1) {
    auto micro_second1 = base::GetTimestamp();
    uint64_t tp_sum1 = parameterServiceImpl.GetThroughputCounterSum();
    std::this_thread::sleep_for(std::chrono::seconds(1));
    auto micro_second2 = base::GetTimestamp();
    uint64_t tp_sum2 = parameterServiceImpl.GetThroughputCounterSum();
    double tps = (tp_sum2 - tp_sum1) * 1.0 / (micro_second2 - micro_second1);
    printf("throughput %.4f Mkv/s\n", tps);
  }
  return 0;
}
