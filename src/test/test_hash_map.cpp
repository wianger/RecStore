#include <folly/concurrency/ConcurrentHashMap.h>
#include <folly/container/F14Map.h>
#include <omp.h>

#include <chrono>
#include <iostream>
#include <thread>
#include <unordered_map>

// using dict_type = folly::F14FastMap<
//       uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
//       folly::f14::DefaultAlloc<std::pair<uint64_t const, uint64_t>>>;

using dict_type = folly::ConcurrentHashMap<uint64_t, uint64_t>;

// using dict_type = std::unordered_map<uint64_t, uint64_t>;

int main() {
  dict_type myMap;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < 10; ++i) {
    myMap.insert(i, i);
    // myMap[i] = i;
  }

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

  omp_set_num_threads(36);

#pragma omp parallel
  {
    int avg_shard = (myMap.get_num_shards() - 1) / omp_get_num_threads() + 1;
    int thread_id = omp_get_thread_num();

    std::cout << "thread_id " << thread_id << std::endl;
    printf("avg_shard %d\n", avg_shard);

    auto shard_begin = myMap.get_shard(avg_shard * thread_id);
    auto shard_end = myMap.get_shard(avg_shard * (thread_id + 1));

    if (shard_begin == shard_end) {
      printf("thread %d shard_begin == shard_end\n", thread_id);
    }

    for (; shard_begin != shard_end; ++shard_begin) {
      auto &it = *shard_begin;
      if (thread_id == 0)
        printf("T%d %lu %lu\n", thread_id, it.first, it.second);
    }
  }

  return 0;
}