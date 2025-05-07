#pragma once
#include "base/array.h"
#include "parameters.h"
#include "flatc.h"

#include <string>
#include <vector>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <random>

template<class T>
class Provider{
public:
    virtual void PutTask(T) = 0;
    virtual T GetTask() = 0;
    virtual bool ExistTask() = 0;
    virtual int LeftTask() = 0;
    virtual long long GetTID() = 0;
    virtual void ImporvePriority(T) = 0;
};

template<class T>
class FIFOProvider : public Provider<T>{
public:
    FIFOProvider(){
        task_id = 0;
    }

    void PutTask(T task) override {
        lock_.lock();
        tasklist.push_back(task);
        lock_.unlock();
    }

    T GetTask() override {
        std::lock_guard<std::mutex> _(lock_);
        if(tasklist.size() != 0){
            T temp = tasklist.front(); 
            tasklist.erase(tasklist.begin());  
            return temp;
        }
        return nullptr;
    }

    bool ExistTask() override {
        return tasklist.size() != 0;
    }

    int LeftTask() override {
        return tasklist.size();
    }

    long long GetTID() override {
        return task_id;
    }

    void ImporvePriority(T task) override {
        return;
    }

private:
    std::vector<T> tasklist;
    std::atomic<long long> task_id;
    std::mutex lock_;
};

template<class T>
class PriorityProvider : public Provider<T>{
public:
    PriorityProvider(int level_ = 7){
        task_id = 0;
        last_update = -1;
        accumulate = -1;
        level = level_;
        for(int i = 0; i < level; i++){
            std::vector<T> temp;
            tasklist.push_back(temp);
            probilaty[i+1] = (float)(i + 1);
        }
    }

    void PutTask(T task) override{
        lock_.lock();
        if(priority_map.find(task) == priority_map.end()){
            priority_map[task] = 1;
        }
        tasklist[priority_map[task] - 1].push_back(task);
        lock_.unlock();
    }

    T GetTask() override {
        std::lock_guard<std::mutex> _(lock_);
        if(LeftTask() != 0){
            int pos = PosChoose();
            if(pos != -1 && tasklist[pos].size() != 0){
                T temp = tasklist[pos].front(); 
                tasklist[pos].erase(tasklist[pos].begin());  
                return temp;
            }
        }
        return nullptr;
    }
     
    bool ExistTask() override {
        return LeftTask() != 0;
    }
     
    int LeftTask() override {
        int sum_ = 0;
        for(auto i = tasklist.begin(); i != tasklist.end(); i++){
            sum_ += i->size();
        }
        return sum_;
    }

    long long GetTID() override {
        return task_id;
    }

    int PosChoose() {
        std::vector<float> temp;
        float sum = 0;
        for(int i = 0; i < tasklist.size(); i++){
            // if(tasklist[level-1-i].size() != 0){
            //     return level - 1 - i;
            // }

            if(tasklist[i].size() != 0){
                temp.push_back(probilaty[i+1] * tasklist[i].size());
                sum += probilaty[i+1] * tasklist[i].size();
            }else{
                temp.push_back(0.0);
            }
        }

        if(sum == 0){
            return -1;
        }

        for(auto i = temp.begin(); i!= temp.end(); i++){
            (*i) /= sum;
        }

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        float randomFloat = dis(gen);
        float accu = 0.0;
        int pos = -1;
        for(int i = 0; i < level; i++){
            accu += temp[i];
            if(accu >= randomFloat){
                pos = i;
                break;
            }
        }
        return pos;
    }

    void ImporvePriority(T task) override {
        lock_.lock();
        if(priority_map.find(task) == priority_map.end()){
            priority_map[task] = 1;
        }
        if (priority_map[task] < level)
            priority_map[task] += 1;

        
        std::cout << task << " priority has been improved to level " << priority_map[task] << std::endl;

        bool minus = true;
        for(auto i = priority_map.begin(); i!=priority_map.end(); i++){
            if(i->second == 1){
                minus = false;
                break;
            }
        }

        if(minus){
            for(auto i = priority_map.begin(); i!=priority_map.end(); i++){
                i->second -= 1;
            }
            std::cout << "All minus 1" << std::endl;
        }
        
        lock_.unlock();
    }

private:
    std::atomic<long long> task_id;
    std::mutex lock_;
    int level;
    std::unordered_map<T, int> priority_map;
    std::vector<std::vector<T>> tasklist;
    std::unordered_map<int, float> probilaty; 
    std::atomic<int> last_update;
    std::atomic<int> accumulate;
};