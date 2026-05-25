#include <folly/init/Init.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "base/array.h"
#include "ps/rdma/allshards_ps_client.h"
#include "ps/rdma/petps_client.h"

DEFINE_int32(num_shards, 1, "number of RDMA RC shards");
DEFINE_int32(iterations, 100, "number of iterations per round");
DEFINE_int32(rounds, 5, "number of measured rounds");
DEFINE_int32(warmup_rounds, 1, "number of warmup rounds before measurement");
DEFINE_int32(batch_keys, 16, "number of keys per request");
DEFINE_int32(get_ratio, 95, "GET percentage for mixed operation");
DEFINE_int32(async_depth, 1, "outstanding async GET depth for async_depth op");
DEFINE_string(op, "all", "operation: all|put|get|async_get|async_depth|mixed");
DEFINE_string(report_mode,
              "summary",
              "benchmark output mode: summary|per_round|both");
DECLARE_int32(value_size);

namespace {

constexpr const char* kTransportName = "RDMA_RC";

struct BenchmarkInput {
  std::vector<std::uint64_t> keys;
  std::vector<std::vector<float>> values;
  base::ConstArray<std::uint64_t> key_array;
};

std::vector<std::uint64_t> MakeKeys(int batch_keys) {
  CHECK_GT(batch_keys, 0) << "--batch_keys must be positive";
  std::vector<std::uint64_t> keys;
  keys.reserve(static_cast<std::size_t>(batch_keys));
  for (int i = 0; i < batch_keys; ++i) {
    keys.push_back(static_cast<std::uint64_t>(1000001 + i));
  }
  return keys;
}

std::vector<std::vector<float>>
MakeValues(const std::vector<std::uint64_t>& keys) {
  const int dim = FLAGS_value_size / sizeof(float);
  CHECK_GT(dim, 0) << "--value_size must be at least sizeof(float)";
  std::vector<std::vector<float>> values;
  values.reserve(keys.size());
  for (auto key : keys) {
    std::vector<float> row;
    row.reserve(static_cast<std::size_t>(dim));
    for (int col = 0; col < dim; ++col) {
      row.push_back(static_cast<float>(key * 10 + col));
    }
    values.push_back(std::move(row));
  }
  return values;
}

BenchmarkInput MakeInput() {
  BenchmarkInput input;
  input.keys      = MakeKeys(FLAGS_batch_keys);
  input.values    = MakeValues(input.keys);
  input.key_array = base::ConstArray<std::uint64_t>(input.keys);
  return input;
}

bool ShouldRunOp(const std::string& requested, const std::string& op) {
  return requested == "all" || requested == op;
}

bool ShouldPrintPerRound(const std::string& mode) {
  return mode == "per_round" || mode == "both";
}

bool ShouldPrintSummary(const std::string& mode) {
  return mode == "summary" || mode == "both";
}

bool MixedIterationIsGet(int iteration) {
  if (FLAGS_get_ratio == 100) {
    return true;
  }
  if (FLAGS_get_ratio == 0) {
    return false;
  }
  const int put_ratio = 100 - FLAGS_get_ratio;
  const int period    = std::max(1, 100 / put_ratio);
  return (iteration + 1) % period != 0;
}

void ValidateFlags() {
  CHECK_GT(FLAGS_num_shards, 0);
  CHECK_GT(FLAGS_iterations, 0);
  CHECK_GE(FLAGS_warmup_rounds, 0);
  CHECK_GT(FLAGS_rounds, 0);
  CHECK_GT(FLAGS_batch_keys, 0);
  CHECK(FLAGS_op == "all" || FLAGS_op == "put" || FLAGS_op == "get" ||
        FLAGS_op == "async_get" || FLAGS_op == "async_depth" ||
        FLAGS_op == "mixed")
      << "Invalid --op: " << FLAGS_op;
  CHECK_GT(FLAGS_async_depth, 0);
  CHECK_GE(FLAGS_get_ratio, 0);
  CHECK_LE(FLAGS_get_ratio, 100);
  CHECK(FLAGS_report_mode == "summary" || FLAGS_report_mode == "per_round" ||
        FLAGS_report_mode == "both")
      << "Invalid --report_mode: " << FLAGS_report_mode;
}

void MaybePrintPerRound(
    const std::string& op,
    bool is_warmup,
    int round,
    int warmup_rounds,
    int measure_rounds,
    int64_t elapsed_us) {
  if (!ShouldPrintPerRound(FLAGS_report_mode)) {
    return;
  }
  std::cout << "transport=" << kTransportName << " op=" << op
            << " phase=" << (is_warmup ? "warmup" : "measure") << " round="
            << (is_warmup ? (round + 1) : (round - warmup_rounds + 1)) << "/"
            << (is_warmup ? warmup_rounds : measure_rounds)
            << " elapsed_us=" << elapsed_us << std::endl;
}

double PercentileUs(std::vector<int64_t> samples, double ratio) {
  CHECK(!samples.empty());
  CHECK_GE(ratio, 0.0);
  CHECK_LE(ratio, 1.0);
  std::sort(samples.begin(), samples.end());
  const std::size_t idx = static_cast<std::size_t>(std::min<int64_t>(
      samples.size() - 1,
      static_cast<int64_t>(std::ceil(ratio * samples.size()) - 1)));
  return static_cast<double>(samples[idx]);
}

void PrintSummary(const std::string& op,
                  const std::string& phase,
                  const std::vector<int64_t>& elapsed_us_samples,
                  int iterations_per_round,
                  std::size_t keys_per_iteration) {
  if (elapsed_us_samples.empty()) {
    return;
  }

  const double total_us = std::accumulate(
      elapsed_us_samples.begin(), elapsed_us_samples.end(), 0.0);
  const double mean_us =
      total_us / static_cast<double>(elapsed_us_samples.size());
  const double p50_us       = PercentileUs(elapsed_us_samples, 0.50);
  const double p95_us       = PercentileUs(elapsed_us_samples, 0.95);
  const double p99_us       = PercentileUs(elapsed_us_samples, 0.99);
  const double total_rounds = static_cast<double>(elapsed_us_samples.size());
  const double ops_per_sec =
      (static_cast<double>(iterations_per_round) * total_rounds) /
      (total_us / 1e6);
  const double key_ops_per_sec =
      (static_cast<double>(iterations_per_round) *
       static_cast<double>(keys_per_iteration) * total_rounds) /
      (total_us / 1e6);

  std::cout << "transport=" << kTransportName << " op=" << op << " phase="
            << phase << " summary rounds=" << elapsed_us_samples.size()
            << " iterations=" << iterations_per_round << " batch_keys="
            << FLAGS_batch_keys << " elapsed_us_mean=" << mean_us
            << " elapsed_us_p50=" << p50_us << " elapsed_us_p95=" << p95_us
            << " elapsed_us_p99=" << p99_us << " ops_per_sec=" << ops_per_sec
            << " key_ops_per_sec=" << key_ops_per_sec << std::endl;
}

template <typename IterationFn>
void RunOperation(const std::string& op,
                  IterationFn&& run_iteration,
                  std::size_t keys_per_iteration) {
  const int total_rounds = FLAGS_warmup_rounds + FLAGS_rounds;
  std::vector<int64_t> warmup_samples_us;
  std::vector<int64_t> measure_samples_us;
  warmup_samples_us.reserve(static_cast<std::size_t>(FLAGS_warmup_rounds));
  measure_samples_us.reserve(static_cast<std::size_t>(FLAGS_rounds));

  for (int round = 0; round < total_rounds; ++round) {
    const bool is_warmup = round < FLAGS_warmup_rounds;
    const auto start     = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < FLAGS_iterations; ++iteration) {
      run_iteration(iteration);
    }
    const auto end = std::chrono::steady_clock::now();
    const int64_t elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    (is_warmup ? warmup_samples_us : measure_samples_us).push_back(elapsed_us);
    MaybePrintPerRound(
        op, is_warmup, round, FLAGS_warmup_rounds, FLAGS_rounds, elapsed_us);
  }

  if (ShouldPrintSummary(FLAGS_report_mode)) {
    PrintSummary(
        op, "warmup", warmup_samples_us, FLAGS_iterations, keys_per_iteration);
    PrintSummary(op,
                 "measure",
                 measure_samples_us,
                 FLAGS_iterations,
                 keys_per_iteration);
  }
}

class BenchmarkClient {
public:
  explicit BenchmarkClient(int num_shards) {
    CHECK_GT(num_shards, 0);
    if (num_shards == 1) {
      shard_clients_.push_back(
          std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, 0));
      client_ = shard_clients_.front().get();
    } else {
      std::vector<BaseParameterClient*> raw_clients;
      raw_clients.reserve(static_cast<std::size_t>(num_shards));
      for (int shard = 0; shard < num_shards; ++shard) {
        shard_clients_.push_back(
            std::make_unique<petps::PetPSClient>("127.0.0.1", 1234, shard));
        raw_clients.push_back(shard_clients_.back().get());
      }
      multi_client_ = std::make_unique<AllShardsParameterClientWrapper>(
          raw_clients, num_shards);
      client_ = multi_client_.get();
    }
    client_->InitThread();
  }

  BaseParameterClient* get() const { return client_; }

private:
  std::vector<std::unique_ptr<petps::PetPSClient>> shard_clients_;
  std::unique_ptr<AllShardsParameterClientWrapper> multi_client_;
  BaseParameterClient* client_ = nullptr;
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  ValidateFlags();

  BenchmarkInput input = MakeInput();
  BenchmarkClient benchmark_client(FLAGS_num_shards);
  BaseParameterClient* client = benchmark_client.get();
  CHECK_NE(client, nullptr);

  CHECK_EQ(client->PutParameter(input.keys, input.values), 0)
      << "initial RDMA RC PutParameter failed";

  if (ShouldRunOp(FLAGS_op, "put")) {
    RunOperation(
        "put",
        [&](int iteration) {
          CHECK_EQ(client->PutParameter(input.keys, input.values), 0)
              << "RDMA RC put failed at iteration=" << iteration;
        },
        input.keys.size());
  }

  if (ShouldRunOp(FLAGS_op, "get")) {
    const int dim = FLAGS_value_size / sizeof(float);
    std::vector<float> output(
        input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
    RunOperation(
        "get",
        [&](int iteration) {
          int rpc_id =
              client->GetParameter(input.key_array, output.data(), false, 0);
          client->WaitRPCFinish(rpc_id);
          client->RevokeRPCResource(rpc_id);
          (void)iteration;
        },
        input.keys.size());
  }

  if (ShouldRunOp(FLAGS_op, "async_get")) {
    const int dim = FLAGS_value_size / sizeof(float);
    std::vector<float> output(
        input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
    RunOperation(
        "async_get",
        [&](int iteration) {
          int rpc_id =
              client->GetParameter(input.key_array, output.data(), true, 0);
          client->WaitRPCFinish(rpc_id);
          client->RevokeRPCResource(rpc_id);
          (void)iteration;
        },
        input.keys.size());
  }

  if (ShouldRunOp(FLAGS_op, "async_depth")) {
    const int dim = FLAGS_value_size / sizeof(float);
    std::vector<std::vector<float>> outputs;
    outputs.reserve(static_cast<std::size_t>(FLAGS_async_depth));
    for (int i = 0; i < FLAGS_async_depth; ++i) {
      outputs.emplace_back(
          input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
    }
    std::vector<int> rpc_ids(static_cast<std::size_t>(FLAGS_async_depth), 0);
    RunOperation(
        "async_depth" + std::to_string(FLAGS_async_depth),
        [&](int iteration) {
          for (int i = 0; i < FLAGS_async_depth; ++i) {
            rpc_ids[static_cast<std::size_t>(i)] = client->GetParameter(
                input.key_array,
                outputs[static_cast<std::size_t>(i)].data(),
                true,
                0);
          }
          for (int rpc_id : rpc_ids) {
            client->WaitRPCFinish(rpc_id);
            client->RevokeRPCResource(rpc_id);
          }
          (void)iteration;
        },
        input.keys.size() * static_cast<std::size_t>(FLAGS_async_depth));
  }

  if (ShouldRunOp(FLAGS_op, "mixed")) {
    const int dim = FLAGS_value_size / sizeof(float);
    std::vector<float> output(
        input.keys.size() * static_cast<std::size_t>(dim) + 1, 0.0f);
    RunOperation(
        "mixed_get" + std::to_string(FLAGS_get_ratio) + "_put" +
            std::to_string(100 - FLAGS_get_ratio),
        [&](int iteration) {
          if (MixedIterationIsGet(iteration)) {
            int rpc_id =
                client->GetParameter(input.key_array, output.data(), false, 0);
            client->WaitRPCFinish(rpc_id);
            client->RevokeRPCResource(rpc_id);
          } else {
            CHECK_EQ(client->PutParameter(input.keys, input.values), 0)
                << "RDMA RC mixed put failed at iteration=" << iteration;
          }
        },
        input.keys.size());
  }

  return 0;
}
