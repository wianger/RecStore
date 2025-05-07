#include <iostream>
#include <string>

int main(int argc, char** argv){
 int deviceCount;
 cudaError_t err = cudaGetDeviceCount(&deviceCount);
 std::cout << "device Count = " << deviceCount;
 if (err != cudaSuccess)
 {
     std::cout <<  "main, cudaGetDeviceCount:" << cudaGetErrorString(err) <<std::endl << std::flush;
     throw std::string("Unexpected error: ") + cudaGetErrorString(err);
 }
 return 0;
}