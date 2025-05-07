#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
using namespace std;

void read() {
  std::ifstream file("/tmp/ssd", std::ifstream::binary);
  float f;
  cout << "1111111111" << endl;
  while (file >> f) std::cout << f << std::endl;
  cout << "1111111111" << endl;
  return;
}

void write() {
  std::vector<float> float_vec;
  char *buffer = (char *)malloc(512 * 100);
  memset(buffer, 0, 512 * 100);
  for (int i = 0; i < 100; i++) {
    float_vec.assign(32, i);
    memcpy(buffer + i * 512, float_vec.data(), 128);
  }

  std::ofstream file("/tmp/write_to_ssd", std::ios::binary);
  file.write(buffer, 512 * 100);
  return;
}

// sudo nvme write /dev/nvme1n1  --start-block=0 --block-count=100
// --data-size=51200  --data /tmp/write_to_ssd sudo nvme read /dev/nvme1n1
// --start-block=0 --block-count=100 --data-size=51200  --data /tmp/ssd

int main() { write(); }