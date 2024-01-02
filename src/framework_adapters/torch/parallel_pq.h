#pragma once
#include <array>
#include <iostream>
#include <unordered_map>

namespace recstore {

namespace {
template <typename T>
struct Node {
  T data;
  int queue_no;
  Node* prev;
  Node* next;

  Node(const T& value, int queue_no)
      : data(value), queue_no(queue_no), prev(nullptr), next(nullptr) {}
};
}  // namespace

template <typename T>
class DoublyLinkedList {
 private:
  Node<T>* head;
  Node<T>* tail;
  std::atomic_long size_;
  int queue_no;

 public:
  DoublyLinkedList(int queue_no)
      : head(nullptr), tail(nullptr), size_(0), queue_no(queue_no) {}

  void insert(Node<T>* newNode) {
    if (!head) {
      head = tail = newNode;
    } else {
      tail->next = newNode;
      newNode->prev = tail;
      tail = newNode;
    }
    size_++;
  }

  void remove(Node<T>* nodeToRemove) {
    if (nodeToRemove->prev) {
      nodeToRemove->prev->next = nodeToRemove->next;
    } else {
      // if head node
      head = nodeToRemove->next;
    }
    if (nodeToRemove->next) {
      nodeToRemove->next->prev = nodeToRemove->prev;
    } else {
      // if tail node
      tail = nodeToRemove->prev;
    }
    delete nodeToRemove;
    size_--;
  }

  Node<T>* pop() {
    Node<T>* nodeToRemove = head;
    head = head->next;
    head->prev = nullptr;
    size_--;
    return nodeToRemove;
  }

  Node<T>* top() { return head; }

  size_t size() const { return size_; }

  bool empty() const { return size_ == 0; }

  void print() {
    Node<T>* current = head;
    while (current) {
      std::cout << current->data << " ";
      current = current->next;
    }
    std::cout << std::endl;
  }

  void CheckConsistency() {
    Node<T>* current = head;
    while (current) {
      if (current->prev) CHECK_EQ(current->prev->next, current);
      CHECK_EQ(current->data->Priority(), queue_no);
      current = current->next;
    }
  }
};

template <typename T>
class ParallelPq {
  constexpr static int kMaxPriority = 1000;
  // priority ranges from 0~<kMaxPriority-1>

 public:
  ParallelPq() {
    for (int i = 0; i < kMaxPriority; i++) {
      qs_[i] = new DoublyLinkedList<T>(i);
    }
  }

  void push(const T& value) {
    base::LockGuard _(lock_);
    push_inner(value);
  }

  void PushOrUpdate(const T& value) {
    base::LockGuard _(lock_);
    if (hashTable.find(value) == hashTable.end()) {
      push_inner(value);
    } else {
      adjustPriority(value);
    }
  }

  void adjustPriority(const T& value) {
    Node<T>* node = hashTable[value];
    int new_priority = value->Priority();

    qs_[node->queue_no]->remove(node);
    node->queue_no = new_priority;
    qs_[new_priority]->insert(node);
  }

  void ForDebug(const std::string& head) {
    base::LockGuard _(lock_);

    // for (auto each : data_) {
    //   if (each->GetID() == 1718) {
    //     LOG(INFO) << head << " find 1718 " << each->ToString() << ".\n top is
    //     "
    //               << top()->ToString();
    //     CheckConsistency();
    //     return;
    //   }
    // }
    CheckConsistency();
  }

  void CheckConsistency(const std::string& hint = "") {
    for (int i = 0; i < kMaxPriority; i++) {
      qs_[i]->CheckConsistency();
    }
  }

  T top() const {
    for (int i = min_priority_now_; i < kMaxPriority; i++) {
      if (qs_[i]->empty()) {
        min_priority_now_ = i + 1;
      } else {
        return qs_[i]->top()->data;
      }
    }
  }

  size_t size() const {
    size_t size = 0;
    for (int i = min_priority_now_; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        size += qs_[i]->size();
      }
    }
    return size;
  }

  bool empty() const {
    for (int i = min_priority_now_; i < kMaxPriority; i++) {
      if (!qs_[i]->empty()) {
        return false;
      }
    }
    return true;
  }

  T pop() {
    for (int i = min_priority_now_; i < kMaxPriority; i++) {
      if (qs_[i]->empty()) {
        min_priority_now_ = i + 1;
      } else {
        Node<T>* head = qs_[i]->pop();
        hashTable.erase(head->data);
        delete head;
        return head->data;
      }
    }
    LOG(FATAL) << "empty queue";
    return nullptr;
  }

 private:
  void push_inner(const T& value) {
    int priority = value->Priority();
    Node<T>* newNode = new Node<T>(value, priority);
    qs_[priority]->insert(newNode);

    // update hashtable
    hashTable[value] = newNode;
  }

  std::array<DoublyLinkedList<T>*, kMaxPriority> qs_;
  mutable int min_priority_now_ = 0;
  std::unordered_map<T, Node<T>*> hashTable;
  mutable base::SpinLock lock_;
};
}  // namespace recstore
