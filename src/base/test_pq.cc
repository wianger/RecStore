#include <algorithm>
#include <random>
#include <vector>

#include "common.h"
#include "log.h"
#include "pq.h"

class CustomElement {
 public:
  CustomElement(int value) : value(value) {}

  int Priority() const { return value; }

  std::string ToString() { return std::to_string(value); }

  int GetID() const { return value; }

  int value;
};

struct CompareCustomElement {
  bool operator()(const CustomElement* a, const CustomElement* b) const {
    return a->Priority() > b->Priority();
  }
};

int main() {
  base::CustomPriorityQueue<CustomElement*, CompareCustomElement> priorityQueue;

  std::vector<CustomElement*> elements;
  std::mt19937 rng(static_cast<unsigned>(100));
  std::uniform_int_distribution<int> dist(1, 100);

  for (int i = 0; i < 20; ++i) {
    elements.emplace_back(new CustomElement(dist(rng)));
  }

  for (auto elem : elements) {
    LOG(INFO) << "Push " << elem->value;
    priorityQueue.push(elem);
    LOG(INFO) << priorityQueue.ToString();
    priorityQueue.CheckConsistency();
  }

  LOG(INFO) << "sort elements";
  std::sort(elements.begin(), elements.end(),
            [](const CustomElement* a, const CustomElement* b) {
              return a->Priority() > b->Priority();
            });

  LOG(INFO) << "sort done";
  LOG(INFO) << priorityQueue.ToString();

  // 确保队列中的最大元素是预期的最大元素
  CHECK_EQ(
      priorityQueue.top()->Priority(),
      (*std::min_element(elements.begin(), elements.end(),
                         [](const CustomElement* a, const CustomElement* b) {
                           return a->Priority() < b->Priority();
                         }))
          ->Priority());

  // 测试 pop
  LOG(INFO) << "before pop";
  priorityQueue.CheckConsistency();
  priorityQueue.pop();
  LOG(INFO) << "after pop";
  priorityQueue.CheckConsistency();

  CHECK_EQ(
      priorityQueue.top()->Priority(),
      (*std::min_element(elements.begin(), elements.end() - 1,
                         [](const CustomElement* a, const CustomElement* b) {
                           return a->Priority() < b->Priority();
                         }))
          ->Priority());

  // 测试 adjustPriority
  CustomElement* elementToAdjust = elements[2];

  for (int i = 0; i < 100; i++) {
    int newPriority = i;
    LOG(INFO) << "from " << elementToAdjust->value << " -> " << newPriority;
    elementToAdjust->value = newPriority;

    LOG(INFO) << priorityQueue.ToString();

    priorityQueue.PushOrUpdate(elementToAdjust);
    LOG(INFO) << "after adjust";
    LOG(INFO) << priorityQueue.ToString();
    priorityQueue.CheckConsistency();
  }

  std::cout << "All tests passed!" << std::endl;

  for (int _ = 0; _ < 1e6; _++) {
    if (priorityQueue.empty()) {
      elements.push_back(new CustomElement(dist(rng)));
      priorityQueue.push(elements.back());
    } else {
      if (dist(rng) > 50) {
        elements.push_back(new CustomElement(dist(rng)));
        priorityQueue.push(elements.back());
      } else {
        priorityQueue.CheckConsistency();

        auto select = folly::Random::rand32() % elements.size();
        elements[select]->value = dist(rng);
        priorityQueue.PushOrUpdate(elements[select]);

        priorityQueue.CheckConsistency();
      }
    }
    priorityQueue.CheckConsistency();
  }

  return 0;
}