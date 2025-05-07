#include "base/bind_core.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <vector>

namespace base {
int global_socket_id = 0;

std::vector<std::vector<int>> parse_numa_nodes() {
  FILE* pipe = popen("lscpu", "r");
  if (!pipe) {
    LOG(FATAL) << "Failed to run lscpu command" << std::endl;
  }

  char buffer[2000];
  std::string result = "";
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
    result += buffer;
  }
  pclose(pipe);

  std::istringstream ss(result);
  std::string line;
  std::vector<std::string> numa_lines;
  while (std::getline(ss, line)) {
    if (line.find("NUMA node") != std::string::npos &&
        line.find("CPU(s):") != std::string::npos) {
      numa_lines.push_back(line);
    }
  }

  std::vector<std::vector<int>> core_table;

  for (size_t i = 0; i < numa_lines.size(); ++i) {
    core_table.push_back(std::vector<int>());
    std::string& numa_line = numa_lines[i];
    size_t pos = numa_line.find("CPU(s):");
    if (pos != std::string::npos) {
      std::string cpus = numa_line.substr(pos + 7);
      std::istringstream cpu_stream(cpus);
      std::string cpu_range;
      while (std::getline(cpu_stream, cpu_range, ',')) {
        size_t dash_pos = cpu_range.find('-');
        if (dash_pos != std::string::npos) {
          int start = std::stoi(cpu_range.substr(0, dash_pos));
          int end = std::stoi(cpu_range.substr(dash_pos + 1));
          for (int cpu = start; cpu <= end; ++cpu) {
            core_table[i].push_back(cpu);
          }
        } else {
          core_table[i].push_back(std::stoi(cpu_range));
        }
      }
    }
  }
  return core_table;
}

std::vector<std::vector<int>> core_table = parse_numa_nodes();

void auto_bind_core() {
  static std::atomic<int> cur_id{0};
  int core_idx = cur_id.fetch_add(1);
  LOG(WARNING) << "bind to core " << core_table[global_socket_id][core_idx];
  bind_core(core_table[global_socket_id][core_idx]);
}
}  // namespace base