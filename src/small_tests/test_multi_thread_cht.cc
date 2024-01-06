#include <iostream>
#include <thread>
#include <unordered_map>

#include "folly/concurrency/ConcurrentHashMap.h"

folly::ConcurrentHashMap<int64_t, int64_t> hashTable;

void insertIntoHashTable(int tid) {
  hashTable.assign(tid, tid);
  for (int _ = 0; _ < 10; _++) {
    // int64_t key = 100;
    // int64_t value = key + tid;
    // auto [no_use, success] = hashTable.insert_or_assign(key, value);
    // assert(success);
    // CHECK(success);
    auto iter = hashTable.assign_if_equal(tid, tid, tid + 1);
    LOG(WARNING) << "tid" << tid << " " << iter.has_value() << "|"
                 << hashTable[tid] << std::endl;
  }
}

int main() {
  int nr_thread = 32;

  std::vector<std::thread> threads;

  for (int i = 0; i < nr_thread; i++) {
    threads.emplace_back(&insertIntoHashTable, i);
  }
  for (int i = 0; i < nr_thread; i++) {
    threads[i].join();
  }

  //   for (const auto& pair : hashTable) {
  //     std::cout << "Key: " << pair.first << ", Value: " << pair.second
  //               << std::endl;
  //   }

  return 0;
}
