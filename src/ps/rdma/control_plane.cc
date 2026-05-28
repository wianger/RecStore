#include "ps/rdma/control_plane.h"

#include <arpa/inet.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

#include "base/json.h"

namespace petps {
namespace {

std::string HexEncode(const void* data, std::size_t size) {
  static constexpr char kHex[] = "0123456789abcdef";
  const auto* bytes            = static_cast<const std::uint8_t*>(data);
  std::string out;
  out.resize(size * 2);
  for (std::size_t i = 0; i < size; ++i) {
    out[i * 2]     = kHex[(bytes[i] >> 4) & 0x0f];
    out[i * 2 + 1] = kHex[bytes[i] & 0x0f];
  }
  return out;
}

int HexValue(char ch) {
  if (ch >= '0' && ch <= '9') {
    return ch - '0';
  }
  if (ch >= 'a' && ch <= 'f') {
    return 10 + (ch - 'a');
  }
  if (ch >= 'A' && ch <= 'F') {
    return 10 + (ch - 'A');
  }
  return -1;
}

RawVerbsNodeMeta DecodeMetaHex(const std::string& hex) {
  if (hex.size() != sizeof(RawVerbsNodeMeta) * 2) {
    throw std::runtime_error("invalid RawVerbsNodeMeta hex size");
  }
  RawVerbsNodeMeta meta{};
  auto* bytes = reinterpret_cast<std::uint8_t*>(&meta);
  for (std::size_t i = 0; i < sizeof(RawVerbsNodeMeta); ++i) {
    const int high = HexValue(hex[i * 2]);
    const int low  = HexValue(hex[i * 2 + 1]);
    if (high < 0 || low < 0) {
      throw std::runtime_error("invalid RawVerbsNodeMeta hex character");
    }
    bytes[i] = static_cast<std::uint8_t>((high << 4) | low);
  }
  return meta;
}

std::string EncodeMetaHex(const RawVerbsNodeMeta& meta) {
  return HexEncode(&meta, sizeof(meta));
}

std::string EndpointString(const RdmaControlPlaneEndpoint& endpoint) {
  return endpoint.host + ":" + std::to_string(endpoint.port);
}

void CloseFd(int* fd) {
  if (fd != nullptr && *fd >= 0) {
    close(*fd);
    *fd = -1;
  }
}

void SetSocketTimeouts(int fd, int timeout_ms) {
  if (timeout_ms <= 0) {
    return;
  }
  struct timeval timeout;
  timeout.tv_sec  = timeout_ms / 1000;
  timeout.tv_usec = (timeout_ms % 1000) * 1000;
  setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
  setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
}

void WriteAll(int fd, const std::string& data) {
  std::size_t offset = 0;
  while (offset < data.size()) {
    const ssize_t written =
        send(fd, data.data() + offset, data.size() - offset, 0);
    if (written < 0) {
      throw std::runtime_error(
          "control-plane send failed: " + std::string(std::strerror(errno)));
    }
    if (written == 0) {
      throw std::runtime_error("control-plane send returned zero");
    }
    offset += static_cast<std::size_t>(written);
  }
}

std::string ReadLine(int fd) {
  std::string line;
  char ch = '\0';
  while (true) {
    const ssize_t rc = recv(fd, &ch, 1, 0);
    if (rc < 0) {
      throw std::runtime_error(
          "control-plane recv failed: " + std::string(std::strerror(errno)));
    }
    if (rc == 0) {
      break;
    }
    if (ch == '\n') {
      break;
    }
    line.push_back(ch);
  }
  return line;
}

json ReadJsonLine(int fd) {
  const std::string line = ReadLine(fd);
  if (line.empty()) {
    throw std::runtime_error("control-plane received empty request");
  }
  return json::parse(line);
}

void WriteJsonLine(int fd, const json& payload) {
  WriteAll(fd, payload.dump() + "\n");
}

int ConnectToEndpoint(const RdmaControlPlaneEndpoint& endpoint) {
  struct addrinfo hints {};
  hints.ai_family         = AF_UNSPEC;
  hints.ai_socktype       = SOCK_STREAM;
  struct addrinfo* result = nullptr;
  const std::string port  = std::to_string(endpoint.port);
  const int rc =
      getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error(
        "control-plane getaddrinfo failed for " + EndpointString(endpoint) +
        ": " + gai_strerror(rc));
  }

  int fd = -1;
  for (struct addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
    fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) {
      continue;
    }
    SetSocketTimeouts(fd, endpoint.timeout_ms);
    if (connect(fd, addr->ai_addr, addr->ai_addrlen) == 0) {
      break;
    }
    CloseFd(&fd);
  }
  freeaddrinfo(result);

  if (fd < 0) {
    throw std::runtime_error(
        "control-plane connect failed for " + EndpointString(endpoint));
  }
  return fd;
}

int BindAndListen(const RdmaControlPlaneEndpoint& endpoint) {
  struct addrinfo hints {};
  hints.ai_family         = AF_UNSPEC;
  hints.ai_socktype       = SOCK_STREAM;
  hints.ai_flags          = AI_PASSIVE;
  struct addrinfo* result = nullptr;
  const std::string port  = std::to_string(endpoint.port);
  const int rc =
      getaddrinfo(endpoint.host.c_str(), port.c_str(), &hints, &result);
  if (rc != 0) {
    throw std::runtime_error(
        "control-plane getaddrinfo failed for " + EndpointString(endpoint) +
        ": " + gai_strerror(rc));
  }

  int fd = -1;
  for (struct addrinfo* addr = result; addr != nullptr; addr = addr->ai_next) {
    fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
    if (fd < 0) {
      continue;
    }
    int reuse = 1;
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
    if (bind(fd, addr->ai_addr, addr->ai_addrlen) == 0 &&
        listen(fd, 128) == 0) {
      break;
    }
    CloseFd(&fd);
  }
  freeaddrinfo(result);

  if (fd < 0) {
    throw std::runtime_error(
        "control-plane bind/listen failed for " + EndpointString(endpoint));
  }
  return fd;
}

json MakeError(const std::string& error) {
  return json{{"ok", false}, {"error", error}};
}

json MakeOk() { return json{{"ok", true}}; }

void WakeListener(const RdmaControlPlaneEndpoint& endpoint) {
  try {
    const int fd = ConnectToEndpoint(endpoint);
    close(fd);
  } catch (...) {
  }
}

} // namespace

RdmaControlPlaneClient::RdmaControlPlaneClient(
    RdmaControlPlaneEndpoint endpoint)
    : endpoint_(std::move(endpoint)) {}

void RdmaControlPlaneClient::PublishMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    const RawVerbsNodeMeta& meta) const {
  const json request = {
      {"type", "publish_meta"},
      {"publisher_node_id", publisher_node_id},
      {"publisher_lane", publisher_lane},
      {"receiver_node_id", receiver_node_id},
      {"receiver_lane", receiver_lane},
      {"meta_hex", EncodeMetaHex(meta)},
  };
  const int fd = ConnectToEndpoint(endpoint_);
  try {
    WriteJsonLine(fd, request);
    const json response = ReadJsonLine(fd);
    if (!response.value("ok", false)) {
      throw std::runtime_error(
          response.value("error", std::string("publish_meta failed")));
    }
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
}

RawVerbsNodeMeta RdmaControlPlaneClient::GetMeta(
    int publisher_node_id,
    int publisher_lane,
    int receiver_node_id,
    int receiver_lane,
    int timeout_ms) const {
  const json request = {
      {"type", "get_meta"},
      {"publisher_node_id", publisher_node_id},
      {"publisher_lane", publisher_lane},
      {"receiver_node_id", receiver_node_id},
      {"receiver_lane", receiver_lane},
      {"timeout_ms", timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms},
  };
  const int fd = ConnectToEndpoint(endpoint_);
  RawVerbsNodeMeta meta{};
  try {
    WriteJsonLine(fd, request);
    const json response = ReadJsonLine(fd);
    if (!response.value("ok", false)) {
      throw std::runtime_error(
          response.value("error", std::string("get_meta failed")));
    }
    meta = DecodeMetaHex(response.at("meta_hex").get<std::string>());
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
  return meta;
}

void RdmaControlPlaneClient::PublishServerReady(int server_id) const {
  const json request = {
      {"type", "server_ready"},
      {"server_id", server_id},
  };
  const int fd = ConnectToEndpoint(endpoint_);
  try {
    WriteJsonLine(fd, request);
    const json response = ReadJsonLine(fd);
    if (!response.value("ok", false)) {
      throw std::runtime_error(
          response.value("error", std::string("server_ready failed")));
    }
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
}

void RdmaControlPlaneClient::WaitServer(int server_id, int timeout_ms) const {
  const json request = {
      {"type", "wait_server"},
      {"server_id", server_id},
      {"timeout_ms", timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms},
  };
  const int fd = ConnectToEndpoint(endpoint_);
  try {
    WriteJsonLine(fd, request);
    const json response = ReadJsonLine(fd);
    if (!response.value("ok", false)) {
      throw std::runtime_error(
          response.value("error", std::string("wait_server failed")));
    }
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
}

void RdmaControlPlaneClient::WaitServerReady(int num_servers,
                                             int timeout_ms) const {
  const json request = {
      {"type", "wait_server_ready"},
      {"num_servers", num_servers},
      {"timeout_ms", timeout_ms > 0 ? timeout_ms : endpoint_.timeout_ms},
  };
  const int fd = ConnectToEndpoint(endpoint_);
  try {
    WriteJsonLine(fd, request);
    const json response = ReadJsonLine(fd);
    if (!response.value("ok", false)) {
      throw std::runtime_error(
          response.value("error", std::string("wait_server_ready failed")));
    }
  } catch (...) {
    close(fd);
    throw;
  }
  close(fd);
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
  if (accept_thread_.joinable()) {
    return;
  }
  stop_requested_.store(false, std::memory_order_relaxed);
  listen_fd_     = BindAndListen(endpoint_);
  accept_thread_ = std::thread(&RdmaControlPlaneServer::ListenLoop, this);
}

void RdmaControlPlaneServer::Stop() {
  stop_requested_.store(true, std::memory_order_relaxed);
  {
    std::lock_guard<std::mutex> guard(mu_);
    cv_.notify_all();
  }
  WakeListener(endpoint_);
  CloseFd(&listen_fd_);
  if (accept_thread_.joinable()) {
    accept_thread_.join();
  }
  for (auto& thread : handler_threads_) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  handler_threads_.clear();
}

void RdmaControlPlaneServer::ListenLoop() {
  while (!stop_requested_.load(std::memory_order_relaxed)) {
    int fd = accept(listen_fd_, nullptr, nullptr);
    if (fd < 0) {
      if (stop_requested_.load(std::memory_order_relaxed)) {
        return;
      }
      continue;
    }
    SetSocketTimeouts(fd, endpoint_.timeout_ms);
    handler_threads_.emplace_back(
        &RdmaControlPlaneServer::HandleConnection, this, fd);
  }
}

void RdmaControlPlaneServer::HandleConnection(int fd) {
  try {
    const json request     = ReadJsonLine(fd);
    const std::string type = request.value("type", std::string());
    if (type == "publish_meta") {
      const MetaKey key{
          request.at("publisher_node_id").get<int>(),
          request.at("publisher_lane").get<int>(),
          request.at("receiver_node_id").get<int>(),
          request.at("receiver_lane").get<int>(),
      };
      const RawVerbsNodeMeta meta =
          DecodeMetaHex(request.at("meta_hex").get<std::string>());
      {
        std::lock_guard<std::mutex> guard(mu_);
        metadata_[key] = meta;
      }
      cv_.notify_all();
      WriteJsonLine(fd, MakeOk());
    } else if (type == "get_meta") {
      const MetaKey key{
          request.at("publisher_node_id").get<int>(),
          request.at("publisher_lane").get<int>(),
          request.at("receiver_node_id").get<int>(),
          request.at("receiver_lane").get<int>(),
      };
      const int timeout_ms = request.value("timeout_ms", endpoint_.timeout_ms);
      std::unique_lock<std::mutex> lock(mu_);
      const bool ready =
          cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return stop_requested_.load(std::memory_order_relaxed) ||
                   metadata_.find(key) != metadata_.end();
          });
      if (!ready) {
        WriteJsonLine(
            fd,
            MakeError("get_meta timeout key=" +
                      std::to_string(key.publisher_node_id) + ":" +
                      std::to_string(key.publisher_lane) + "->" +
                      std::to_string(key.receiver_node_id) + ":" +
                      std::to_string(key.receiver_lane)));
      } else if (stop_requested_.load(std::memory_order_relaxed)) {
        WriteJsonLine(fd, MakeError("control-plane stopping"));
      } else {
        const RawVerbsNodeMeta meta = metadata_.at(key);
        WriteJsonLine(
            fd, json{{"ok", true}, {"meta_hex", EncodeMetaHex(meta)}});
      }
    } else if (type == "server_ready") {
      {
        std::lock_guard<std::mutex> guard(mu_);
        ready_servers_.insert(request.at("server_id").get<int>());
      }
      cv_.notify_all();
      WriteJsonLine(fd, MakeOk());
    } else if (type == "wait_server") {
      const int server_id  = request.at("server_id").get<int>();
      const int timeout_ms = request.value("timeout_ms", endpoint_.timeout_ms);
      std::unique_lock<std::mutex> lock(mu_);
      const bool ready =
          cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return stop_requested_.load(std::memory_order_relaxed) ||
                   ready_servers_.find(server_id) != ready_servers_.end();
          });
      if (!ready) {
        WriteJsonLine(fd,
                      MakeError("wait_server timeout server_id=" +
                                std::to_string(server_id)));
      } else if (stop_requested_.load(std::memory_order_relaxed)) {
        WriteJsonLine(fd, MakeError("control-plane stopping"));
      } else {
        WriteJsonLine(fd, MakeOk());
      }
    } else if (type == "wait_server_ready") {
      const int num_servers = request.at("num_servers").get<int>();
      const int timeout_ms  = request.value("timeout_ms", endpoint_.timeout_ms);
      std::unique_lock<std::mutex> lock(mu_);
      const bool ready =
          cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] {
            return stop_requested_.load(std::memory_order_relaxed) ||
                   static_cast<int>(ready_servers_.size()) >= num_servers;
          });
      if (!ready) {
        WriteJsonLine(fd,
                      MakeError("wait_server_ready timeout ready=" +
                                std::to_string(ready_servers_.size()) + "/" +
                                std::to_string(num_servers)));
      } else if (stop_requested_.load(std::memory_order_relaxed)) {
        WriteJsonLine(fd, MakeError("control-plane stopping"));
      } else {
        WriteJsonLine(fd, MakeOk());
      }
    } else if (type == "shutdown") {
      WriteJsonLine(fd, MakeOk());
      stop_requested_.store(true, std::memory_order_relaxed);
      cv_.notify_all();
      CloseFd(&listen_fd_);
    } else {
      WriteJsonLine(fd, MakeError("unknown request type: " + type));
    }
  } catch (const std::exception& e) {
    try {
      WriteJsonLine(fd, MakeError(e.what()));
    } catch (...) {
    }
  }
  close(fd);
}

} // namespace petps
