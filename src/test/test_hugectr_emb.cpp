#include "framework/hugectr/op_hugectr.h"
#include <iostream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <cuda_runtime_api.h>

#include "../third_party/HugeCTR/HugeCTR/include/tensor2.hpp"

/* build:
g++ -std=c++17 -O2 \
    src/test/test_hugectr_emb.cpp \
    src/framework/op.cc \
    src/framework/hugectr/op_hugectr.cc \
    -o build/hugectr_test \
    -I./src \
    -I/usr/local/cuda/include \
    -I./third_party/HugeCTR/HugeCTR/include \
    -I./third_party/HugeCTR/gpu_cache/include \
    -I./third_party/HugeCTR/HugeCTR \
    $(python3 -c 'import torch.utils.cpp_extension as e; print(" ".join(["-I" + p for p in e.include_paths()]))') \
    -L./third_party/HugeCTR/_build/lib \
    -L/usr/local/cuda/lib64 \
    $(python3 -c 'import torch.utils.cpp_extension as e; print(" ".join(["-L" + p for p in e.library_paths()]))') \
    -ltorch -ltorch_cpu -ltorch_python -lc10 -lcublas -lcudart
*/

/* Run:
LD_LIBRARY_PATH=$(python3 -c 'import torch.utils.cpp_extension as e; print(":".join(e.library_paths()))'):./third_party/HugeCTR/_build/lib ./build/hugectr_test
*/

class RawPtrBuffer : public HugeCTR::TensorBuffer2 {
private:
  void* ptr_;

public:
  RawPtrBuffer(void* ptr) : ptr_(ptr) {}

  void* get_ptr() override { return ptr_; }
  
  bool allocated() const override { return ptr_ != nullptr; }
};
void check_cuda_error(cudaError_t err) {
  if (err != cudaSuccess) {
    throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(err)));
  }
}

template <typename T>
void print_vector(const std::string& name, const std::vector<T>& vec, size_t limit = 5) {
  std::cout << name << " (first " << limit << " elements): [";
  for (size_t i = 0; i < vec.size() && i < limit; ++i) {
    std::cout << vec[i] << (i < vec.size() - 1 && i < limit - 1 ? ", " : "");
  }
  std::cout << "]" << std::endl;
}

int main() {
  const int batch_size = 16;
  const int emb_dim = base::EMBEDDING_DIMENSION_D;

  std::cout << "--- HugeCTR Interface Test (Final API Fix) ---" << std::endl;
  std::cout << "Batch Size (L): " << batch_size << std::endl;
  std::cout << "Embedding Dim (D): " << emb_dim << std::endl;
  std::cout << "----------------------------------------------" << std::endl;

  void* d_keys_ptr = nullptr;
  void* d_values_ptr = nullptr;
  void* d_grads_ptr = nullptr;

  try {
    std::vector<long long> h_keys(batch_size);
    for (int i = 0; i < batch_size; ++i) { h_keys[i] = 1000 + i; }
    std::vector<float> h_grads(batch_size * emb_dim);
    for (int i = 0; i < batch_size * emb_dim; ++i) { h_grads[i] = static_cast<float>(i % 100) * 0.01f; }
    print_vector("Host Keys", h_keys);

    size_t keys_size_bytes = h_keys.size() * sizeof(long long);
    size_t grads_size_bytes = h_grads.size() * sizeof(float);
    size_t values_size_bytes = batch_size * emb_dim * sizeof(float);
    check_cuda_error(cudaMalloc(&d_keys_ptr, keys_size_bytes));
    check_cuda_error(cudaMalloc(&d_values_ptr, values_size_bytes));
    check_cuda_error(cudaMalloc(&d_grads_ptr, grads_size_bytes));

    check_cuda_error(cudaMemcpy(d_keys_ptr, h_keys.data(), keys_size_bytes, cudaMemcpyHostToDevice));
    check_cuda_error(cudaMemcpy(d_grads_ptr, h_grads.data(), grads_size_bytes, cudaMemcpyHostToDevice));
    
    HugeCTR::Tensor2<long long> d_keys({(size_t)batch_size}, 
                                      std::make_shared<RawPtrBuffer>(d_keys_ptr));
    HugeCTR::Tensor2<float> d_values({(size_t)batch_size, (size_t)emb_dim},
                                     std::make_shared<RawPtrBuffer>(d_values_ptr));
    HugeCTR::Tensor2<float> d_grads({(size_t)batch_size, (size_t)emb_dim},
                                    std::make_shared<RawPtrBuffer>(d_grads_ptr));

    std::cout << "\n--- Testing emb_read_hugectr ---" << std::endl;
    recstore::framework::emb_read_hugectr(d_keys, d_values);
    std::cout << "emb_read_hugectr called successfully." << std::endl;

    std::vector<float> h_values(batch_size * emb_dim);
    check_cuda_error(cudaMemcpy(h_values.data(), d_values_ptr, values_size_bytes, cudaMemcpyDeviceToHost));
    std::cout << "Result of read copied back to host." << std::endl;
    print_vector("Host Values (from Read)", h_values, emb_dim);

    std::cout << "\n--- Testing emb_update_hugectr ---" << std::endl;
    recstore::framework::emb_update_hugectr(d_keys, d_grads);
    std::cout << "emb_update_hugectr called successfully." << std::endl;

    std::cout << "\n----------------------------------------------" << std::endl;
    std::cout << "Test completed successfully!" << std::endl;
    std::cout << "----------------------------------------------" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "An error occurred: " << e.what() << std::endl;
    if (d_keys_ptr) cudaFree(d_keys_ptr);
    if (d_values_ptr) cudaFree(d_values_ptr);
    if (d_grads_ptr) cudaFree(d_grads_ptr);
    return 1;
  }

  cudaFree(d_keys_ptr);
  cudaFree(d_values_ptr);
  cudaFree(d_grads_ptr);

  return 0;
}
