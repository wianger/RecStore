#include <folly/container/F14Map.h>
#include <folly/concurrency/ConcurrentHashMap.h>
#include <thread>
#include <iostream>
#include <unordered_map>
#include <chrono>

// using dict_type = folly::F14FastMap<
//       uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
//       folly::f14::DefaultAlloc<std::pair<uint64_t const, uint64_t>>>;

// using dict_type = folly::ConcurrentHashMap<uint64_t, uint64_t>;

using dict_type = std::unordered_map<uint64_t, uint64_t>;

int main() {
    dict_type myMap;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 12000000; ++i) {
        // myMap.insert(i, i);
        myMap[i] = i;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}