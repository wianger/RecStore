#include <folly/portability/GTest.h>

#include <vector>

#include "RawGDWrapper.h"
#include "base/base.h"
#include "base/cuda.h"
#include "base/timer.h"
using gpu_direct_ssd::RawGDWrapper;

DEFINE_int32(embedding_dimension, 32, "");
DEFINE_int32(key_space_M, 1, "key space in millions");
DEFINE_int32(query_count, 100, "# of query embs in one round");
DEFINE_int32(run_time, 60, "benchmark time in seconds");

TEST(RawGDSSD, test) {
  const int emb_dim = FLAGS_embedding_dimension;
  uint64_t test_key_capability = FLAGS_key_space_M * 1e6;
  const int query_count = FLAGS_query_count;

  RawGDWrapper<uint64_t> BaMWrapper(test_key_capability, emb_dim);
  BaMWrapper.InitFakeDB();

  std::vector<uint64_t> index(query_count);
  float *d_output_buffer;
  float *h_output_buffer;
  XMH_CUDA_CHECK(cudaMalloc(&d_output_buffer, query_count * emb_dim * sizeof(float)));
  XMH_CUDA_CHECK(cudaMallocHost(&h_output_buffer, query_count * emb_dim * sizeof(float)));
  uint64_t *d_index;
  XMH_CUDA_CHECK(cudaMalloc(&d_index, query_count * sizeof(uint64_t)));

  auto start_time = std::chrono::steady_clock::now();
  for (int _ = 0; _ < 30; _++) {
    auto now_time = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now_time - start_time).count() >
        FLAGS_run_time)
      break;

    for (int i = 0; i < query_count; i++) {
      index[i] = folly::Random::rand64() % test_key_capability;
    }
    XMH_CUDA_CHECK(cudaMemcpy(d_index, index.data(), query_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

    XMH_CUDA_CHECK(cudaGetLastError());
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    xmh_nv::Event before;
    BaMWrapper.Query(d_output_buffer, d_index, query_count, 0);
    xmh_nv::Event after;
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    double elapsed_us = after - before;

    XMH_CUDA_CHECK(cudaMemcpy(h_output_buffer, d_output_buffer,
                              query_count * emb_dim * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    for (int i = 0; i < query_count; i++) {
      for (int j = 0; j < emb_dim; j++) {
        CHECK_NEAR(h_output_buffer[i * emb_dim + j], index[i], 1e-6);
      }
    }

    xmh::Timer::ManualRecordNs("GPUDirect query", elapsed_us * 1e3);
  }

  double elapsed_us = xmh::Timer::ManualQuery("GPUDirect query") / 1e3;
  uint64_t ios = query_count;
  uint64_t data = query_count * sizeof(float) * emb_dim;
  double iops = ((double)ios) / (elapsed_us / 1e6);
  double bandwidth =
      (((double)data) / (elapsed_us / 1e6)) / (1024ULL * 1024ULL * 1024ULL);
  std::cout << std::dec << "Elapsed Time (us): " << elapsed_us
            << "\tNumber of Read Ops: " << ios << "\tData Size (bytes): " << data
            << std::endl;
  std::cout << std::dec << "Read Ops/sec: " << iops
            << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

  xmh::Reporter::Report();
  LOG(INFO) << "Perf successfully";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  xmh::Reporter::StartReportThread(1000);
  return RUN_ALL_TESTS();
}