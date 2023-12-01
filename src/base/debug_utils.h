#pragma once 

#include <iostream>
#include <vector>
#include <sstream>
namespace base {
template <typename T>
std::string ToString(const std::vector<T> & vec){
  std::ostringstream oss;
  for (auto each: vec){
    oss << each << " ";
  }
  return oss.str();
}
}