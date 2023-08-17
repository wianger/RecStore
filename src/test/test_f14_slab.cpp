#include <folly/container/F14Map.h>
#include <thread>
#include <iostream>

using dict_type = folly::F14FastMap<
      uint64_t, uint64_t, std::hash<uint64_t>, std::equal_to<uint64_t>,
      folly::f14::DefaultAlloc<std::pair<uint64_t const, uint64_t>>>;

dict_type hash_table_;

uint64_t KEY_SIZE = 1000000;
uint64_t THREAD_NUM = 2;
std::atomic<int> now_thread(0);

void test_map(int thread_id){
    now_thread.fetch_add(1);
    while(now_thread.load() != THREAD_NUM);
    for(int i = 0; i < KEY_SIZE; i++){
        bool ret = false;
        while(ret == false){
            auto [a, b] = hash_table_.insert({thread_id * KEY_SIZE + i, thread_id * KEY_SIZE + i});
            ret = b;
        }
    }
    for(int i = 0; i < KEY_SIZE; i++){
        if(hash_table_.at(thread_id * KEY_SIZE + i) != thread_id * KEY_SIZE + i){
            throw std::runtime_error("error");
        }
    }
}

int main(){
    std::vector<std::thread> threads;
    for(int i = 0; i < THREAD_NUM; i++){
        threads.emplace_back(test_map, i);
    }
    for(auto &t : threads){
        t.join();
    }
    std::cout << "finish" << std::endl;
    return 0;
}

