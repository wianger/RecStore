#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ps/rdma/raw_verbs_transport.h"

namespace grpc {
class Server;
}

namespace petps {

class RdmaControlPlaneService;

struct RdmaControlPlaneEndpoint {
  std::string host = "127.0.0.1";
  int port         = 25100;
  int timeout_ms   = 30000;
};

class RdmaControlPlaneClient {
public:
  explicit RdmaControlPlaneClient(RdmaControlPlaneEndpoint endpoint);

  void PublishMeta(int publisher_node_id,
                   int publisher_lane,
                   int receiver_node_id,
                   int receiver_lane,
                   const RawVerbsNodeMeta& meta) const;
  RawVerbsNodeMeta
  GetMeta(int publisher_node_id,
          int publisher_lane,
          int receiver_node_id,
          int receiver_lane,
          int timeout_ms = -1) const;
  void PublishServerReady(int server_id) const;
  void WaitServer(int server_id, int timeout_ms = -1) const;
  void WaitServerReady(int num_servers, int timeout_ms = -1) const;

private:
  RdmaControlPlaneEndpoint endpoint_;
};

class RdmaControlPlaneServer {
public:
  explicit RdmaControlPlaneServer(RdmaControlPlaneEndpoint endpoint);
  ~RdmaControlPlaneServer();

  void Start();
  void Stop();

private:
  friend class RdmaControlPlaneService;

  struct MetaKey {
    int publisher_node_id = 0;
    int publisher_lane    = 0;
    int receiver_node_id  = 0;
    int receiver_lane     = 0;

    bool operator==(const MetaKey& other) const {
      return publisher_node_id == other.publisher_node_id &&
             publisher_lane == other.publisher_lane &&
             receiver_node_id == other.receiver_node_id &&
             receiver_lane == other.receiver_lane;
    }
  };

  struct MetaKeyHash {
    std::size_t operator()(const MetaKey& key) const;
  };

  RdmaControlPlaneEndpoint endpoint_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::unordered_map<MetaKey, RawVerbsNodeMeta, MetaKeyHash> metadata_;
  std::unordered_set<int> ready_servers_;
  std::atomic<bool> stop_requested_{false};
  std::unique_ptr<RdmaControlPlaneService> service_;
  std::unique_ptr<grpc::Server> server_;
};

} // namespace petps
