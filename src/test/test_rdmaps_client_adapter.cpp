#include <gtest/gtest.h>

#include <vector>

#include "framework/common/ps_client_config_adapter.h"
#include "ps/client_factory.h"
#include "ps/rdma/rdma_ps_client_adapter.h"

namespace recstore {

TEST(RDMAPSClientAdapterTest, FactoryCreatesRdmaClientAndSupportsTableInit) {
  json config = {
      {"cache_ps",
       {{"ps_type", "RDMA"},
        {
            "base_kv_config",
            {{"value", {{"default_value_size_hint", 16}}}},
        },
        {"num_threads", 1}}},
      {"client", {{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}},
      {"distributed_client",
       {{"num_shards", 1},
        {"hash_method", "city_hash"},
        {"max_keys_per_request", 64},
        {"servers",
         json::array(
             {{{"host", "127.0.0.1"}, {"port", 25000}, {"shard", 0}}})}}},
  };

  auto client =
      CreatePSClient(ResolvePSClientOptionsFromFrameworkConfig(config));
  ASSERT_NE(client, nullptr);
  EXPECT_NE(dynamic_cast<RDMAPSClientAdapter*>(client.get()), nullptr);
}

} // namespace recstore
