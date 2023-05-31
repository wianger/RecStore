#include <folly/init/Init.h>

#include <vector>

#include "base/array.h"
#include "folly/Random.h"
#include "naiveKVell.h"

using ssdps::NaiveArraySSD;

#define XMH_CUDA_CHECK(val) \
  { xmh_nv::cuda_check_((val), __FILE__, __LINE__); }

namespace xmh_nv {

class CudaException : public std::runtime_error {
 public:
  CudaException(const std::string &what) : runtime_error(what) {}
};

inline void cuda_check_(cudaError_t val, const char *file, int line) {
  if (val != cudaSuccess) {
    throw CudaException(std::string(file) + ":" + std::to_string(line) +
                        ": CUDA error " + std::to_string(val) + ": " +
                        cudaGetErrorString(val));
  }
}
}  // namespace xmh_nv

DEFINE_int32(embedding_dimension, 32, "");
DEFINE_int32(key_space_M, 1, "key space in millions");
DEFINE_int32(query_count, 100, "# of query embs in one round");
DEFINE_int32(run_time, 60, "benchmark time in seconds");

int main(int argc, char **argv) {
  folly::init(&argc, &argv);
  xmh::Reporter::StartReportThread(2000);

  const int emb_dim = FLAGS_embedding_dimension;
  uint64_t test_key_capability = FLAGS_key_space_M * 1e6;
  const int query_count = FLAGS_query_count;

  NaiveArraySSD<uint64_t> ssd(emb_dim * 4, test_key_capability);

  // Init DB
  // {
  //   std::vector<uint64_t> keys;
  //   std::vector<float> values;
  //   keys.reserve(test_key_capability);
  //   values.reserve(test_key_capability * emb_dim);
  //   for (int i = 0; i < test_key_capability; i++) {
  //     keys.push_back(i);
  //     for (int j = 0; j < emb_dim; j++) values.push_back(i);
  //   }
  //   ConstArray<uint64_t> keys_array(keys);
  //   LOG(INFO) << "bulk load finished";
  // }

  void *p;
  XMH_CUDA_CHECK(
      cudaMallocHost(&p, query_count * sizeof(uint64_t), cudaHostAllocDefault));
  MutableArray<uint64_t> test_get_keys_array((uint64_t *)p, query_count);
  uint64_t *device_keys;
  XMH_CUDA_CHECK(cudaMalloc(&device_keys, query_count * sizeof(uint64_t)));

  ConstArray<uint64_t> index_array;

  XMH_CUDA_CHECK(cudaMallocHost(&p, query_count * emb_dim * sizeof(float),
                                cudaHostAllocDefault));
  MutableArray<float> test_get_values((float *)p, query_count * emb_dim);

  XMH_CUDA_CHECK(cudaMalloc(&p, query_count * emb_dim * sizeof(float)));
  MutableArray<float> d_test_get_values((float *)p, query_count * emb_dim);

  auto start_time = std::chrono::steady_clock::now();
  // while (1) {
  for (int _ = 0; _ < 1000; _++) {
    auto now_time = std::chrono::steady_clock::now();
    if (std::chrono::duration_cast<std::chrono::seconds>(now_time - start_time)
            .count() > FLAGS_run_time)
      break;

    CHECK_EQ(test_get_keys_array.Size(), query_count);
    for (int i = 0; i < query_count; i++) {
      test_get_keys_array[i] = folly::Random::rand64() % test_key_capability;
    }
    XMH_CUDA_CHECK(cudaMemcpy(device_keys, test_get_keys_array.binary_data(),
                              query_count * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
    xmh::Timer timer_kvell("SPDK query");
    {
      xmh::Timer timer_kvell_keyd2h("key_d2h");
      XMH_CUDA_CHECK(cudaMemcpy(test_get_keys_array.binary_data(), device_keys,
                                query_count * sizeof(uint64_t),
                                cudaMemcpyDeviceToHost));
      timer_kvell_keyd2h.end();
    }

    ssd.BatchGet(test_get_keys_array.ToConstArray(), index_array,
                 (void *)test_get_values.Data());
    {
      xmh::Timer timer_kvell_valueh2d("value_h2d");
      XMH_CUDA_CHECK(
          cudaMemcpy(d_test_get_values.Data(), test_get_values.Data(),
                     test_get_values.binary_size(), cudaMemcpyHostToDevice));
      timer_kvell_valueh2d.end();
    }

    timer_kvell.end();

    // for (int i = 0; i < query_count; i++) {
    //   for (int j = 0; j < emb_dim; j++) {
    //     CHECK_NEAR(test_get_keys_array[i], test_get_values[i * emb_dim + j],
    //                1e-6);
    //   }
    // }
    // std::this_thread::sleep_for(std::chrono::seconds(1));
  }
  double elapsed_us = xmh::Timer::ManualQuery("SPDK query") / 1e3;
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