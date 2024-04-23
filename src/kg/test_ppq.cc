#include "parallel_pq.h"

using namespace recstore;

void test_ppq() {
  DoublyLinkedList<int> myList;
  myList.insert(1);
  myList.insert(2);
  myList.insert(3);
  myList.print();

  Node<int>* foundNode = myList.find(2);
  if (foundNode) {
    std::cout << "Found: " << foundNode->data << std::endl;
  } else {
    std::cout << "Not found" << std::endl;
  }

  myList.remove(2);
  myList.print();

  foundNode = myList.find(2);
  if (foundNode) {
    std::cout << "Found: " << foundNode->data << std::endl;
  } else {
    std::cout << "Not found" << std::endl;
  }

  return 0;
}