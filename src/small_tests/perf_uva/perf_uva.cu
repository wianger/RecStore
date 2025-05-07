#include <folly/Random.h>
#include <folly/init/Init.h>

#include "base/cuda.h"
#include "base/cudaEvent.h"
#include "base/timer.h"

DEFINE_int32(emb_dim, 32, "");
DEFINE_int32(key_space_M, 1, "key space in millions");
DEFINE_int32(query_count, 100, "# of query embs in one round");
DEFINE_int32(run_time, 60, "benchmark time in seconds");
DEFINE_string(method, "CPU", "CPU/UVA");

void HostQuery(const uint64_t *d_indices, const float *h_db, const int emb_dim,
               float *d_output_buffer, const int query_count) {
  static uint64_t *h_indices = nullptr;
  static float *h_output_buffer = nullptr;
  if (h_indices == nullptr) {
    XMH_CUDA_CHECK(cudaMallocHost(&h_indices, query_count * sizeof(uint64_t),
                                  cudaHostAllocDefault));
    XMH_CUDA_CHECK(cudaMallocHost(&h_output_buffer,
                                  query_count * emb_dim * sizeof(float),
                                  cudaHostAllocDefault));
  }

  xmh::Timer key_timer("key d2h");
  XMH_CUDA_CHECK(cudaMemcpy(h_indices, d_indices,
                            query_count * sizeof(uint64_t),
                            cudaMemcpyDeviceToHost));
  key_timer.end();

  xmh::Timer cpu_gather_timer("cpu gather");
  for (int i = 0; i < query_count; i++) {
    for (int j = 0; j < emb_dim; j++) {
      h_output_buffer[i * emb_dim + j] = h_db[h_indices[i] * emb_dim + j];
    }
  }
  cpu_gather_timer.end();

  xmh::Timer value_timer("value h2d");
  XMH_CUDA_CHECK(cudaMemcpy(d_output_buffer, h_output_buffer,
                            query_count * emb_dim * sizeof(float),
                            cudaMemcpyHostToDevice));
                            
  value_timer.end();
  return;
}

__global__ static void UVAQueryKernel(const uint64_t *d_indices,
                                      const float *h_db, const int emb_dim,
                                      float *d_output_buffer,
                                      const int query_count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < query_count * emb_dim) {
    uint64_t key_idx = idx / emb_dim;
    int sub_idx = idx % emb_dim;
    d_output_buffer[idx] = h_db[d_indices[key_idx] * emb_dim + sub_idx];
  }
}

void UVAQuery(const uint64_t *d_indices, const float *h_db, const int emb_dim,
              float *d_output_buffer, const int query_count) {
  uint64_t b_size = 128;
  uint64_t numThreads = query_count * emb_dim;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  UVAQueryKernel<<<g_size, b_size>>>(d_indices, h_db, emb_dim, d_output_buffer,
                                     query_count);
  return;
}

__global__ static void CheckKernel(const uint64_t *d_indices, const int emb_dim,
                                   float *d_output_buffer,
                                   const int query_count) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < query_count * emb_dim) {
    uint64_t key_idx = idx / emb_dim;
    int sub_idx = idx % emb_dim;
    if (fabsf(d_output_buffer[idx] - d_indices[key_idx]) > 1e-6) {
      printf("check failed\n");
      assert(0);
    }
  }
}

void Check(const uint64_t *d_indices, const int emb_dim, float *d_output_buffer,
           const int query_count) {
  uint64_t b_size = 128;
  uint64_t numThreads = query_count * emb_dim;
  uint64_t g_size = (numThreads + b_size - 1) / b_size;
  CheckKernel<<<g_size, b_size>>>(d_indices, emb_dim, d_output_buffer,
                                  query_count);
}

int main(int argc, char **argv) {
  folly::init(&argc, &argv);
  const int emb_dim = FLAGS_emb_dim;
  uint64_t test_key_capability = FLAGS_key_space_M * 1e6;
  const int query_count = FLAGS_query_count;

  float *db = nullptr;
  XMH_CUDA_CHECK(cudaMallocHost(&db,
                                test_key_capability * emb_dim * sizeof(float),
                                cudaHostAllocDefault));

  for (int i = 0; i < test_key_capability; i++)
    for (int j = 0; j < emb_dim; j++) {
      db[i * emb_dim + j] = i;
    }

  std::vector<uint64_t> index(query_count);

  float *d_output_buffer;
  XMH_CUDA_CHECK(
      cudaMalloc(&d_output_buffer, query_count * emb_dim * sizeof(float)));
  uint64_t *d_indices;
  XMH_CUDA_CHECK(cudaMalloc(&d_indices, query_count * sizeof(uint64_t)));

  auto start_time = std::chrono::steady_clock::now();
  // while (1) {
  for (int _ = 0; _ < 10000; _++) {
    auto now_time = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now_time - start_time)
            .count() > FLAGS_run_time)
      break;

    for (int i = 0; i < query_count; i++) {
      index[i] = folly::Random::rand64() % test_key_capability;
    }
    XMH_CUDA_CHECK(cudaMemcpy(d_indices, index.data(),
                              query_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));

    XMH_CUDA_CHECK(cudaGetLastError());
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    xmh::cudaEventWrapper before;
    if (FLAGS_method == "CPU") {
      HostQuery(d_indices, db, emb_dim, d_output_buffer, query_count);
    } else if (FLAGS_method == "UVA") {
      UVAQuery(d_indices, db, emb_dim, d_output_buffer, query_count);
    }
    xmh::cudaEventWrapper after;
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    Check(d_indices, emb_dim, d_output_buffer, query_count);
    XMH_CUDA_CHECK(cudaDeviceSynchronize());
    // std::this_thread::sleep_for(std::chrono::seconds(1));
    // LOG(FATAL) << "success one round";
    double elapsed_us = after - before;
    xmh::Timer::ManualRecordNs("query", elapsed_us * 1e3);
  }

  double elapsed_us = xmh::Timer::ManualQuery("query") / 1e3;
  uint64_t ios = query_count;
  uint64_t data = query_count * sizeof(float) * emb_dim;
  double iops = ((double)ios) / (elapsed_us / 1e6);
  double bandwidth =
      (((double)data) / (elapsed_us / 1e6)) / (1024ULL * 1024ULL * 1024ULL);
  std::cout << std::dec << "Elapsed Time (us): " << elapsed_us
            << "\tNumber of Read Ops: " << ios
            << "\tData Size (bytes): " << data << std::endl;
  std::cout << std::dec << "Read Ops/sec: " << iops
            << "\tEffective Bandwidth(GB/S): " << bandwidth << std::endl;

  xmh::Reporter::Report();
  LOG(INFO) << "Perf successfully";
  return 0;
}