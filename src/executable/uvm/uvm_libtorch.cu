#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

#include <iostream>
// 超参数
const int64_t vocab_size = 10000; // 词汇表大小
const int64_t embed_size = 128;   // 嵌入维度
const int64_t batch_size = 32;    // 批处理大小

struct EmbeddingModelClonableImpl : torch::nn::Cloneable<EmbeddingModelClonableImpl>
{
  torch::nn::Embedding embedding;
  torch::Device device;

  EmbeddingModelClonableImpl(int64_t vocab_size, int64_t embed_size, torch::Device device) : embedding(torch::nn::EmbeddingOptions(vocab_size, embed_size)), device(device)
  {
    std::cout << "EmbeddingModelClonableImpl" << std::endl;
    register_module("embedding", embedding);
  }

  void reset() override
  {
    // create a new embedding layer with the similar size to the old one
    torch::nn::Embedding new_embedding = torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, embed_size));
    this->embedding = new_embedding;
    register_module("embedding", embedding);
    // // print the naive pointer of the embedding weight
    // std::cout<< "in reset"<< embedding->weight.device() << std::endl;
    // std::cout<< embedding->weight.data_ptr() << std::endl;
    // for (const auto& child : children_) {
    //   std::cout<< child.key() << " " << child.value().device() << std::endl;
    //   copy->children_[child.key()]->clone_(*child.value(), device);
    // }
    // abort();
    // embedding->to(device);
  }

  torch::Tensor forward(torch::Tensor indices)
  {
    std::cout<< "in forward weight"<< embedding->weight.sizes() << std::endl;
    // indices = indices.to(embedding->weight.device());
    // indices don't need gradient
    // indices.set_requires_grad(false);
    auto res =  embedding->forward(indices);
    std::cout<< "in forward"<< res.device() << std::endl;
    std::cout<< "in forward"<< res.sizes() << std::endl;
    std::cout<< "in forward"<< indices.sizes() << std::endl;

    return res;
  }
};

TORCH_MODULE(EmbeddingModelClonable);


template <typename DataLoader>
void train(int32_t epoch, EmbeddingModelClonable &model, torch::Device device,
           DataLoader &data_loader, torch::optim::Optimizer &optimizer)
{
  model->train();
  for (auto &batch : data_loader)
  {
    auto data = batch.data();
    optimizer.zero_grad();
    std::vector<torch::Device> devices;
    for (int i = 0; i < 4; i++)
    {
      devices.push_back(torch::Device(torch::kCUDA, i));
    }
    auto output = torch::nn::parallel::data_parallel(model, *data, devices, device);
    
    // 显示前向传播的结果
    std::cout << "Embedding output: " << output.sizes() << std::endl;
    std::cout << "Embedding output: " << device << std::endl;

    // 定义一个假的损失函数，进行反向传播和更新
    torch::Tensor target = torch::randn({batch_size, embed_size}).to(device);
    torch::Tensor loss = torch::mse_loss(output, target).to(device);

    // 反向传播
    optimizer.zero_grad();
    std::cout << "after zero_grad" << std::endl;
    loss.backward();
    std::cout << "after backward" << std::endl;
    optimizer.step();
    std::cout << "after step" << std::endl;

    std::cout << "Loss: " << loss.item<float>() << std::endl;
  }
}

int main()
{
  // torch.autograd.set_detect_anomaly(True)
  torch::autograd::AnomalyMode::set_enabled(true);
  // 使用 DataParallel 让模型支持多 GPU
  auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0) : torch::kCPU;

  // 随机生成输入数据
  void *indices_buffers;
  cudaMallocManaged(&indices_buffers, sizeof(torch::kInt64) * batch_size * 100);
  // 生成随机数据
  for (int i = 0; i < batch_size * 100; i++)
  {
    ((int64_t *)indices_buffers)[i] = 0;
  }
  torch::Tensor indices = torch::from_blob(indices_buffers, { 100, batch_size }, torch::kInt64);
  auto dataset = torch::data::datasets::TensorDataset(indices);
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(dataset), torch::data::DataLoaderOptions().batch_size(batch_size));

  // 创建模型
  EmbeddingModelClonable model(vocab_size, embed_size, device);
  model->to(device);
  // print model parameters
  for (const auto &p : model->parameters())
  {
    std::cout<< p.device() << std::endl;
  }

  
  // 定义优化器
  torch::optim::Adam optimizer(model->parameters(),
                               torch::optim::AdamOptions(0.001));

  for (size_t epoch = 1; epoch <= 1; ++epoch)
  {
    train(epoch, model, device, *train_loader, optimizer);
  }

  return 0;
}
