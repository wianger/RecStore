#include <folly/init/Init.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <memory>
#include <vector>

#include "base/array.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"
#include "ps/rdma/rdma_protocol.h"

DECLARE_int32(value_size);
DECLARE_int32(rdma_rc_qps_per_client_per_shard);

namespace {

std::vector<std::vector<float>>
MakeValues(const std::vector<std::uint64_t>& keys, int embedding_dim) {
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row;
    row.reserve(embedding_dim);
    for (int d = 0; d < embedding_dim; ++d) {
      row.push_back(static_cast<float>(key * 10 + d));
    }
    values.push_back(std::move(row));
  }
  return values;
}

void ExpectFlatSlots(const float* buffer,
                     const std::vector<std::vector<float>>& expected,
                     int embedding_dim) {
  for (std::size_t row = 0; row < expected.size(); ++row) {
    for (int col = 0; col < embedding_dim; ++col) {
      EXPECT_FLOAT_EQ(buffer[row * embedding_dim + col], expected[row][col]);
    }
  }
}

petps::PetPSClient& SingleShardClient() {
  static auto* client = []() {
    auto* created = new petps::PetPSClient("127.0.0.1", 1234, 0);
    created->InitThread();
    return created;
  }();
  return *client;
}

} // namespace

TEST(PetPSIntegrationTest, PutGetRoundTripSingleShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {101, 102, 103};
  auto values                     = MakeValues(keys, embedding_dim);

  ASSERT_EQ(client.PutParameter(keys, values), 0);

  void* recv_buffer =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys),
      static_cast<float*>(recv_buffer),
      false);
  client.WaitRPCFinish(rpc_id);

  ExpectFlatSlots(static_cast<float*>(recv_buffer), values, embedding_dim);
  client.RevokeRPCResource(rpc_id);
}

TEST(PetPSIntegrationTest, UpdateGetRoundTripSingleShard) {
  auto& client = SingleShardClient();

  std::vector<std::uint64_t> keys         = {401, 402};
  std::vector<std::vector<float>> initial = {
      {1.0f, 2.0f, 3.0f, 4.0f},
      {5.0f, 6.0f, 7.0f, 8.0f},
  };
  std::vector<std::vector<float>> grads = {
      {0.5f, 1.0f, 1.5f, 2.0f},
      {1.0f, 0.5f, 2.0f, 1.5f},
  };
  std::vector<std::vector<float>> expected = {
      {0.995f, 1.99f, 2.985f, 3.98f},
      {4.99f, 5.995f, 6.98f, 7.985f},
  };

  ASSERT_EQ(client.InitEmbeddingTable("table_update", 128, 4), 0);
  ASSERT_EQ(client.PutParameter(keys, initial), 0);
  ASSERT_EQ(client.UpdateParameter(
                "table_update", base::ConstArray<std::uint64_t>(keys), &grads),
            0);

  std::vector<std::vector<float>> actual;
  ASSERT_EQ(client.GetParameter(base::ConstArray<std::uint64_t>(keys), &actual),
            0);
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t row = 0; row < expected.size(); ++row) {
    ASSERT_EQ(actual[row].size(), expected[row].size());
    for (std::size_t col = 0; col < expected[row].size(); ++col) {
      EXPECT_FLOAT_EQ(actual[row][col], expected[row][col]);
    }
  }
}

TEST(PetPSIntegrationTest, MissingKeysReturnZeroSlots) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {999001, 999002};
  void* recv_buffer =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  int rpc_id = client.GetParameter(
      base::ConstArray<std::uint64_t>(keys),
      static_cast<float*>(recv_buffer),
      false);
  client.WaitRPCFinish(rpc_id);

  const float* values = static_cast<float*>(recv_buffer);
  for (std::size_t i = 0; i < keys.size() * embedding_dim; ++i) {
    EXPECT_FLOAT_EQ(values[i], 0.0f);
  }
  client.RevokeRPCResource(rpc_id);
}

TEST(PetPSIntegrationTest, PutGetRoundTripMultiShard) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);

  auto shard0 = std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, 0);
  auto shard1 = std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, 1);

  shard0->InitThread();
  shard1->InitThread();

  std::vector<BaseParameterClient*> clients = {shard0.get(), shard1.get()};
  AllShardsParameterClientWrapper wrapper(clients, 2);

  std::vector<std::uint64_t> keys = {1, 2, 3, 4, 5, 6};
  auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(wrapper.PutParameter(keys, values), 0);

  std::vector<float> output(keys.size() * embedding_dim + 1, 0.0f);
  int rpc_id = wrapper.GetParameter(
      base::ConstArray<std::uint64_t>(keys), output.data(), false, 0);
  wrapper.WaitRPCFinish(rpc_id);

  ExpectFlatSlots(output.data(), values, embedding_dim);
  wrapper.RevokeRPCResource(rpc_id);
}

TEST(PetPSIntegrationTest, ExhaustedQpPoolFailsLoudly) {
  const int embedding_dim = FLAGS_value_size / sizeof(float);
  auto& client            = SingleShardClient();

  std::vector<std::uint64_t> keys = {701, 702};
  auto values                     = MakeValues(keys, embedding_dim);
  ASSERT_EQ(client.PutParameter(keys, values), 0);

  const int qp_count = FLAGS_rdma_rc_qps_per_client_per_shard;
  ASSERT_GT(qp_count, 0);

  std::vector<void*> recv_buffers;
  std::vector<int> rpc_ids;
  recv_buffers.reserve(static_cast<std::size_t>(qp_count));
  rpc_ids.reserve(static_cast<std::size_t>(qp_count));

  for (int i = 0; i < qp_count; ++i) {
    void* recv_buffer =
        client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
    recv_buffers.push_back(recv_buffer);
    rpc_ids.push_back(client.GetParameter(
        base::ConstArray<std::uint64_t>(keys),
        static_cast<float*>(recv_buffer),
        true));
  }

  void* overflow_recv =
      client.GetReceiveBuffer(client.ResponseBufferBytes(keys.size()));
  EXPECT_THROW(client.GetParameter(base::ConstArray<std::uint64_t>(keys),
                                   static_cast<float*>(overflow_recv),
                                   true),
               std::runtime_error);

  for (int rpc_id : rpc_ids) {
    client.WaitRPCFinish(rpc_id);
    client.RevokeRPCResource(rpc_id);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  return RUN_ALL_TESTS();
}
