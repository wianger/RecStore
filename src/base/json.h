#pragma once
#include <fstream>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

json ParseFile2Json(const std::string &filename) {
  std::ifstream f(filename);
  json data = json::parse(f);
  return data;
}