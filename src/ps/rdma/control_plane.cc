#include "ps/rdma/control_plane.h"

#include <grpcpp/grpcpp.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "rdma_control_plane.grpc.pb.h"

namespace petps {
namespace {

using recstoreps::rdma::GetMetaRequest;
using recstoreps::rdma::GetMetaResponse;
using recstoreps::rdma::ProbeRequest;
using recstoreps::rdma::ProbeResponse;
using recstoreps::rdma::PublishMetaRequest;
using recstoreps::rdma::PublishMetaResponse;
using recstoreps::rdma::PublishServerReadyRequest;
using recstoreps::rdma::PublishServerReadyResponse;
using recstoreps::rdma::RdmaControlPlane;
using recstoreps::rdma::WaitServerReadyRequest;
using recstoreps::rdma::WaitServerReadyResponse;
using recstoreps::rdma::WaitServerRequest;
using recstoreps::rdma::WaitServerResponse;

std::string EndpointString(const RdmaControlPlaneEndpoint& endpoint) {
  return endpoint.host + ":" + std::to_string(endpoint.port);
}

std::chrono::system_clock::time_point DeadlineFromNow(int timeout_ms) {
  return std::chrono::system_clock::now() +
         std::chrono::milliseconds(timeout_ms);
}

std::string GrpcStatusText(const grpc::Status& status) {
  if (status.error_message().empty()) {
    return status.error_code() == grpc::StatusCode::OK
             ? std::string("OK")
             : std::to_string(status.error_code());
  }
  return status.error_message();
}

void ThrowIfNotOk(const grpc::Status& status, const std::string& operation) {
  if (status.ok()) {
    return;
  }
  throw std::runtime_error(
      "control-plane " + operation + " failed: " + GrpcStatusText(status));
}

std::string EncodeMetaBytes(const RawVerbsNodeMeta& meta) {
  return std::string(reinterpret_cast<const char*>(&meta), sizeof(meta));
}

RawVerbsNodeMeta DecodeMetaBytes(const std::string& payload) {
  if (payload.size() != sizeof(RawVerbsNodeMeta)) {
    throw std::runtime_error("invalid RawVerbsNodeMeta payload size: " +
                             std::to_string(payload.size()));
  }
  RawVerbsNodeMeta meta{};
  std::memcpy(&meta, payload.data(), sizeof(meta));
  return meta;
}

grpc::Status MakeDeadlineExceeded(const std::string& message) {
  return grpc::Status(grpc::StatusCode::DEADLINE_EXCEEDED, message);
}

grpc::Status MakeUnavailable(const std::string& message) {
  return grpc::Status(grpc::StatusCode::UNAVAILABLE, message);
}

} // namespace

class RdmaControlPlaneService final : public RdmaControlPlane::Service {
public:
  explicit RdmaControlPlaneService(RdmaControlPlaneServer* owner)
      : owner_(owner) {}

  grpc::Status PublishMeta(grpc::ServerContext*,
                           const PublishMetaRequest* request,
                           PublishMetaResponse*) override {
    if (request->meta().size() != sizeof(RawVerbsNodeMeta)) {
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                          "invalid RawVerbsNodeMeta payload size: " +
                              std::to_string(request->meta().size()));
    }
    const RdmaControlPlaneServer::MetaKey key{
        request->publisher_node_id(),
        request->publisher_lane(),
        request->receiver_node_id(),
        request->receiver_lane(),
    };
    const RawVerbsNodeMeta meta = DecodeMetaBytes(request->meta());
    {
      std::lock_guard<std::mutex> guard(owner_->mu_);
      owner_->metadata_[key] = meta;
    }
    owner_->cv_.notify_all();
    return grpc::Status::OK;
  }

  grpc::Status GetMeta(grpc::ServerContext*,
                       const GetMetaRequest* request,
                       GetMetaResponse* response) override {
    const RdmaControlPlaneServer::MetaKey key{
        request->publisher_node_id(),
        request->publisher_lane(),
        request->receiver_node_id(),
        request->receiver_lane(),
    };
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 owner_->metadata_.find(key) != owner_->metadata_.end();
        });
    if (!ready) {
      return MakeDeadlineExceeded(
          "get_meta timeout key=" + std::to_string(key.publisher_node_id) +
          ":" + std::to_string(key.publisher_lane) + "->" +
          std::to_string(key.receiver_node_id) + ":" +
          std::to_string(key.receiver_lane));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    response->set_meta(EncodeMetaBytes(owner_->metadata_.at(key)));
    return grpc::Status::OK;
  }

  grpc::Status PublishServerReady(grpc::ServerContext*,
                                  const PublishServerReadyRequest* request,
                                  PublishServerReadyResponse*) override {
    {
      std::lock_guard<std::mutex> guard(owner_->mu_);
      owner_->ready_servers_.insert(request->server_id());
    }
    owner_->cv_.notify_all();
    return grpc::Status::OK;
  }

  grpc::Status WaitServer(grpc::ServerContext*,
                          const WaitServerRequest* request,
                          WaitServerResponse*) override {
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 owner_->ready_servers_.find(request->server_id()) !=
                     owner_->ready_servers_.end();
        });
    if (!ready) {
      return MakeDeadlineExceeded("wait_server timeout server_id=" +
                                  std::to_string(request->server_id()));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    return grpc::Status::OK;
  }

  grpc::Status WaitServerReady(grpc::ServerContext*,
                               const WaitServerReadyRequest* request,
                               WaitServerReadyResponse*) override {
    const int timeout_ms =
        request->timeout_ms() > 0
            ? request->timeout_ms()
            : owner_->endpoint_.timeout_ms;
    std::unique_lock<std::mutex> lock(owner_->mu_);
    const bool ready =
        owner_->cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
          return owner_->stop_requested_.load(std::memory_order_relaxed) ||
                 static_cast<int>(owner_->ready_servers_.size()) >=
                     request->num_servers();
        });
    if (!ready) {
      return MakeDeadlineExceeded(
          "wait_server_ready timeout ready=" +
          std::to_string(owner_->ready_servers_.size()) + "/" +
          std::to_string(request->num_servers()));
    }
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    return grpc::Status::OK;
  }

  grpc::Status
  Probe(grpc::ServerContext*, const ProbeRequest*, ProbeResponse*) override {
    if (owner_->stop_requested_.load(std::memory_order_relaxed)) {
      return MakeUnavailable("control-plane stopping");
    }
    return grpc::Status::OK;
  }

private:
  RdmaControlPlaneServer* owner_;
};

RdmaControlPlaneClient::RdmaControlPlaneClient(
    RdmaControlPlaneEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {}

void RdmaControlPlaneClient::PublishMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    const RawVerbsNodeMeta& meta) const {
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  PublishMetaRequest request;
  request.set_publisher_node_id(publisher_node_id);
  request.set_publisher_lane(publisher_lane);
  request.set_receiver_node_id(receiver_node_id);
  request.set_receiver_lane(receiver_lane);
  request.set_meta(EncodeMetaBytes(meta));

  PublishMetaResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(endpoint_.timeout_ms));
  ThrowIfNotOk(stub->PublishMeta(&context, request, &response), "publish_meta");
}

RawVerbsNodeMeta RdmaControlPlaneClient::GetMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  GetMetaRequest request;
  request.set_publisher_node_id(publisher_node_id);
  request.set_publisher_lane(publisher_lane);
  request.set_receiver_node_id(receiver_node_id);
  request.set_receiver_lane(receiver_lane);
  request.set_timeout_ms(effective_timeout_ms);

  GetMetaResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->GetMeta(&context, request, &response), "get_meta");
  return DecodeMetaBytes(response.meta());
}

void RdmaControlPlaneClient::PublishServerReady(int server_id) const {
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  PublishServerReadyRequest request;
  request.set_server_id(server_id);

  PublishServerReadyResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(endpoint_.timeout_ms));
  ThrowIfNotOk(stub->PublishServerReady(&context, request, &response),
               "server_ready");
}

void RdmaControlPlaneClient::WaitServer(int server_id, int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  WaitServerRequest request;
  request.set_server_id(server_id);
  request.set_timeout_ms(effective_timeout_ms);

  WaitServerResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->WaitServer(&context, request, &response), "wait_server");
}

void RdmaControlPlaneClient::WaitServerReady(int num_servers,
                                             int timeout_ms) const {
  const int effective_timeout_ms =
      timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms;
  auto channel = grpc::CreateChannel(
      EndpointString(endpoint_), grpc::InsecureChannelCredentials());
  auto stub = RdmaControlPlane::NewStub(channel);

  WaitServerReadyRequest request;
  request.set_num_servers(num_servers);
  request.set_timeout_ms(effective_timeout_ms);

  WaitServerReadyResponse response;
  grpc::ClientContext context;
  context.set_deadline(DeadlineFromNow(effective_timeout_ms));
  ThrowIfNotOk(stub->WaitServerReady(&context, request, &response),
               "wait_server_ready");
}

RdmaControlPlaneServer::RdmaControlPlaneServer(
    RdmaControlPlaneEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {}

RdmaControlPlaneServer::~RdmaControlPlaneServer() { Stop(); }

std::size_t
RdmaControlPlaneServer::MetaKeyHash::operator()(const MetaKey& key) const {
  std::size_t hash = static_cast<std::size_t>(key.publisher_node_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.publisher_lane);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.receiver_node_id);
  hash = hash * 1315423911u + static_cast<std::size_t>(key.receiver_lane);
  return hash;
}

void RdmaControlPlaneServer::Start() {
  if (server_ != nullptr) {
    return;
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  service_ = std::make_unique<RdmaControlPlaneService>(this);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(
      EndpointString(endpoint_), grpc::InsecureServerCredentials());
  builder.RegisterService(service_.get());
  server_ = builder.BuildAndStart();
  if (server_ == nullptr) {
    throw std::runtime_error("control-plane gRPC server failed to listen on " +
                             EndpointString(endpoint_));
  }
}

void RdmaControlPlaneServer::Stop() {
  if (server_ == nullptr) {
    return;
  }
  stop_requested_.store(true, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> guard(mu_);
    cv_.notify_all();
  }
  server_->Shutdown();
  server_.reset();
  service_.reset();
}

} // namespace petps
