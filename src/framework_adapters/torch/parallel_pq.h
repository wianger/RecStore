#pragma once
#include <array>
#include <iostream>
#include <unordered_map>

#include "base/lock.h"
#include "folly/concurrency/ConcurrentHashMap.h"
#include "src/memory/malloc.h"

namespace recstore {

template <typename T>
class DoublyLinkedList;

template <typename T>
struct Node {
 private:
  T data;
  int queue_priority;
  Node *prev;
  Node *next;

  friend class recstore::DoublyLinkedList<T>;

 public:
  Node(const T &value, int queue_priority)
      : data(value),
        queue_priority(queue_priority),
        prev(nullptr),
        next(nullptr) {}

  T Data() const { return data; }

  void ResetPointer() {
    prev = nullptr;
    next = nullptr;
  }

  int QueuePriority() const { return queue_priority; }

  void SetQueuePriority(int queue_priority) {
    this->queue_priority = queue_priority;
  }
};

template <typename T>
class DoublyLinkedList {
 private:
  Node<T> *head_;
  Node<T> *tail_;
  std::atomic_long size_;
  const int queue_priority_;
  mutable base::SpinLock lock_;

 public:
  DoublyLinkedList(int queue_priority)
      : head_(nullptr),
        tail_(nullptr),
        size_(0),
        queue_priority_(queue_priority) {}

  void insert(Node<T> *newNode) {
    base::LockGuard _(lock_);
    if (!head_) {
      head_ = tail_ = newNode;
    } else {
      tail_->next = newNode;
      newNode->prev = tail_;
      tail_ = newNode;
    }
    size_++;
  }

  void remove(Node<T> *nodeToRemove) {
    base::LockGuard _(lock_);
    CHECK_EQ(nodeToRemove->queue_priority, queue_priority_);
    if (nodeToRemove->prev) {
      nodeToRemove->prev->next = nodeToRemove->next;
    } else {
      // if head_ node
      head_ = nodeToRemove->next;
    }
    if (nodeToRemove->next) {
      nodeToRemove->next->prev = nodeToRemove->prev;
    } else {
      // if tail_ node
      tail_ = nodeToRemove->prev;
    }
    size_--;
  }

  Node<T> *pop() {
    base::LockGuard _(lock_);

    Node<T> *nodeToRemove = head_;
    head_ = head_->next;
    head_->prev = nullptr;
    size_--;
    return nodeToRemove;
  }

  Node<T> *top() { return head_; }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  std::unordered_set<int64_t> CheckConsistency() {
    base::LockGuard _(lock_);
    std::unordered_set<int64_t> id_set;

    Node<T> *current = head_;
    while (current) {
      if (current->prev) CHECK_EQ(current->prev->next, current);
      CHECK_EQ(current->data->Priority(), queue_priority_);
      id_set.insert(current->data->GetID());
      current = current->next;
    }
    return id_set;
  }

  std::string ToString() const {
    base::LockGuard _(lock_);

    std::stringstream ss;
    Node<T> *current = head_;
    int temp = 0;
    while (current) {
      ss << current->data->ToString() << " \n";
      current = current->next;
      temp++;
      if (temp > size_) LOG(FATAL) << "linklist may not be linked properly";
    }
    return ss.str();
  }
};

template <typename T>
class ParallelPq {
  // priority ranges from 0~<kMaxPriority-1>
  constexpr static int kMaxPriority = 1000;

  static constexpr int kInf = std::numeric_limits<int>::max();

  static inline int CastPriorityToQueueNo(int queue_priority) {
    if (queue_priority == kInf) return kMaxPriority - 1;
    CHECK_LT(queue_priority, kMaxPriority - 1)
        << "Please increase kMaxPriority";
    CHECK_GE(queue_priority, 0);
    return queue_priority;
  }

  static inline int CastQueueNoToPriority(int queue_no) {
    if (queue_no == kMaxPriority - 1) return kInf;
    CHECK_LT(queue_no, kMaxPriority - 1) << "Please increase kMaxPriority";
    CHECK_GE(queue_no, 0);
    return queue_no;
  }

 public:
  ParallelPq(int64_t reserve_count = 0) {
    for (int i = 0; i < kMaxPriority; i++) {
      if (i == kMaxPriority - 1)
        qs_[i] = new DoublyLinkedList<T>(kInf);
      else
        qs_[i] = new DoublyLinkedList<T>(i);
    }
    hashTable_.reserve(reserve_count);
  }

  void PushOrUpdate(const T &value) {
    // LOG(INFO) << "PushOrUpdate " << value->GetID();
    if (hashTable_.find(value) == hashTable_.end()) {
      push_inner(value);
    } else {
      adjustPriority(value);
    }
  }

  std::string ToString() const {
    base::LockGuard guard(lock_);
    std::stringstream ss;
    ss << "CustomParallelPriorityQueue:\n";
    if (empty()) {
      ss << "\t\t"
         << "empty\n";
      return ss.str();
    }

    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        ss << "\t\t"
           << "Q" << i << " :" << qs_[i]->ToString() << "\n";
      }
    }
    return ss.str();
  }

  void ForDebug(const std::string &head) {}

  void CheckConsistency(const std::string &hint = "") {
    std::unordered_set<int64_t> id_set;
    for (int i = 0; i < kMaxPriority; i++) {
      auto id_set_per_q = qs_[i]->CheckConsistency();
    }
  }

  bool empty() const {
    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        return false;
      }
    }
    return true;
  }

  // T pop() {
  //   for (int i = 0; i < kMaxPriority; i++) {
  //     if (qs_[i]->empty()) {
  //       ;
  //     } else {
  //       Node<T> *head = qs_[i]->pop();
  //       hashTable_.erase(head->data);
  //       T data = head->data;
  //       // delete head;
  //       recycle_.Recycle(head);
  //       return data;
  //     }
  //   }
  //   LOG(FATAL) << "empty queue";
  //   return nullptr;
  // }

  T top() const {
    base::LockGuard guard(lock_);
    for (int i = 0; i < kMaxPriority; i++) {
      auto *p = qs_[i]->top();
      if (p) return p->Data();
    }
    return nullptr;
  }

  int MinPriority() const {
    for (int i = 0; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) return CastQueueNoToPriority(i);
    }
    return kInf;
  }

  void pop_x(const T &value) {
    LOG(FATAL) << "not USED now";
    // base::LockGuard guard(lock_);
    CHECK(hashTable_.find(value) != hashTable_.end());
    Node<T> *node = hashTable_[value];
    // LOG(ERROR) << "Node<T> * node" << node;
    // LOG(ERROR) << "CastPriorityToQueueNo(node->queue_priority)="
    //            << CastPriorityToQueueNo(node->queue_priority);

    // LOG(ERROR) << "qs_[CastPriorityToQueueNo(node->queue_priority)] = "
    //            << qs_[CastPriorityToQueueNo(node->queue_priority)];
    qs_[CastPriorityToQueueNo(node->QueuePriority())]->remove(node);
    hashTable_.erase(value);
    // delete node;
    recycle_.Recycle(node);
  }

 private:
  void adjustPriority(const T &value) {
    base::LockGuard guard(*value);
    // base::LockGuard guard(lock_);

    // static base::SpinLock adjust_lock;
    // base::LockGuard guard(adjust_lock);

    Node<T> *node;
    do {
      node = hashTable_[value];
      FB_LOG_EVERY_MS(ERROR, 1000) << "node is nullptr";
    } while (!node);

    CHECK(node);
    int new_priority = value->Priority();
    int old_priority = node->QueuePriority();

    // ↓ atomically
    Node<T> *newnode = new Node<T>(value, new_priority);
    qs_[CastPriorityToQueueNo(new_priority)]->insert(newnode);
    qs_[CastPriorityToQueueNo(old_priority)]->remove(node);

    // auto iter = hashTable_.assign_if_equal(value, node, newnode);
    // if (iter.has_value()) recycle_.Recycle(node);

    hashTable_.assign(value, newnode);
    recycle_.Recycle(node);
  }

  void push_inner(const T &value) {
    // base::LockGuard guard(*value);
    // base::LockGuard guard(lock_);

    // static base::SpinLock push_lock;
    // base::LockGuard guard(push_lock);

    int priority = value->Priority();
    Node<T> *newNode = new Node<T>(value, priority);
    // atomically
    auto [_, success] = hashTable_.insert(value, newNode);
    if (success) {
      // LOG(ERROR) << folly::sformat("hashTable_[{}] = {})", value->GetID(),
      //                              newNode);
      qs_[CastPriorityToQueueNo(priority)]->insert(newNode);
    } else {
      delete newNode;
    }
  }

  std::array<DoublyLinkedList<T> *, kMaxPriority> qs_;
  folly::ConcurrentHashMap<T, Node<T> *> hashTable_;

  base::StdDelayedRecycle recycle_;
  mutable std::atomic_int min_priority_now_ = 0;
  mutable base::SpinLock lock_;
};

}  // namespace recstore
