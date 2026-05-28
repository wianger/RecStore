#include "ps/rdma/control_plane.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <gtest/gtest.h>

#include <chrono>
#include <thread>

namespace petps {
namespace {

int AllocateTcpPort() {
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  EXPECT_GE(fd, 0);
  struct sockaddr_in addr {};
  addr.sin_family      = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  addr.sin_port        = 0;
  EXPECT_EQ(bind(fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)), 0);
  socklen_t addr_len = sizeof(addr);
  EXPECT_EQ(getsockname(fd, reinterpret_cast<sockaddr*>(&addr), &addr_len), 0);
  const int port = ntohs(addr.sin_port);
  close(fd);
  return port;
}

TEST(RdmaControlPlaneTest, PublishMetaAndWaitServerReady) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      2000,
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  RawVerbsNodeMeta published{};
  published.node_id   = 7;
  published.base_addr = 0x12345000ULL;
  published.rkey      = 99;

  std::thread publisher([&] {
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    client.PublishMeta(1, 0, 2, 0, published);
    client.PublishServerReady(0);
    client.PublishServerReady(1);
  });

  const RawVerbsNodeMeta fetched = client.GetMeta(1, 0, 2, 0, 1000);
  EXPECT_EQ(fetched.node_id, published.node_id);
  EXPECT_EQ(fetched.base_addr, published.base_addr);
  EXPECT_EQ(fetched.rkey, published.rkey);

  EXPECT_NO_THROW(client.WaitServerReady(2, 1000));
  EXPECT_NO_THROW(client.WaitServer(1, 1000));

  publisher.join();
  server.Stop();
}

TEST(RdmaControlPlaneTest, WaitSpecificServerTimesOut) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  EXPECT_THROW(client.WaitServer(3, 50), std::runtime_error);

  server.Stop();
}

TEST(RdmaControlPlaneTest, WaitServerReadyTimesOut) {
  const RdmaControlPlaneEndpoint endpoint{
      "127.0.0.1",
      AllocateTcpPort(),
      200,
  };
  RdmaControlPlaneServer server(endpoint);
  server.Start();

  RdmaControlPlaneClient client(endpoint);
  EXPECT_THROW(client.WaitServerReady(1, 50), std::runtime_error);

  server.Stop();
}

} // namespace
} // namespace petps
