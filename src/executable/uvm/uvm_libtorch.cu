#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

#include <iostream>

// 定义一个简单的模型，其中包含Embedding层
struct EmbeddingModel : torch::nn::Module {
  // 定义Embedding层
  torch::nn::Embedding embedding;

  // 构造函数
  EmbeddingModel(int64_t vocab_size, int64_t embed_size)
      : embedding(torch::nn::EmbeddingOptions(vocab_size, embed_size)) {
    register_module("embedding", embedding);
  }

  // 前向传播函数
  torch::Tensor forward(torch::Tensor indices) {
    return embedding->forward(indices);
  }
};

template <typename DataLoader>
void train(int32_t epoch, EmbeddingModel& model, torch::Device device,
           DataLoader& data_loader, torch::optim::Optimizer& optimizer,
           size_t dataset_size) {
  model.train();
  size_t batch_idx = 0;
  for (auto& batch : data_loader) {
  }
}

int main() {
  // 超参数
  const int64_t vocab_size = 10000;  // 词汇表大小
  const int64_t embed_size = 128;    // 嵌入维度
  const int64_t batch_size = 32;     // 批处理大小

  // 创建模型
  auto model = std::make_shared<EmbeddingModel>(vocab_size, embed_size);

  // 使用 DataParallel 让模型支持多 GPU
  auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
  torch::DeviceType device_type =
      torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;


  // 随机生成输入数据
  torch::Tensor indices = torch::randint(
      0, vocab_size, {batch_size},
      torch::TensorOptions().dtype(torch::kInt64).device(device));

  auto output= torch::nn::parallel::data_parallel(model, indices);


  // 显示前向传播的结果
  std::cout << "Embedding output: " << output << std::endl;

  // 定义优化器
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.001));

  // 定义一个假的损失函数，进行反向传播和更新
  torch::Tensor target = torch::randn({batch_size, embed_size}).to(device);
  torch::Tensor loss = torch::mse_loss(output, target);

  // 反向传播
  optimizer.zero_grad();
  loss.backward();
  optimizer.step();

  std::cout << "Loss: " << loss.item<float>() << std::endl;

  return 0;
}
