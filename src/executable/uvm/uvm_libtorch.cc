#include <torch/nn/parallel/data_parallel.h>
#include <torch/nn/pimpl.h>
#include <torch/torch.h>

#include <iostream>

#include "base/base.h"
#include "base/timer.h"

DEFINE_int32(nr_gpus, 4, "Number of GPUs to use.");
DEFINE_int32(batch_size, 26 * 1024, "batch size");
DEFINE_int64(vocab_size, 10000, "vocab size");
DEFINE_int32(embed_size, 32, "emb dim");
// DEFINE_int64(sample_number, 10 * 1e6LL, "");
DEFINE_int64(sample_number, 100, "");

struct EmbeddingModelClonableImpl
    : torch::nn::Cloneable<EmbeddingModelClonableImpl> {
  torch::nn::Embedding embedding;
  torch::Device device;

  EmbeddingModelClonableImpl(int64_t vocab_size, int64_t embed_size,
                             torch::Device device)
      : embedding(torch::nn::EmbeddingOptions(vocab_size, embed_size)),
        device(device) {
    std::cout << "EmbeddingModelClonableImpl" << std::endl;
    register_module("embedding", embedding);
  }

  void reset() override {
    // create a new embedding layer with the similar size to the old one
    torch::nn::Embedding new_embedding = torch::nn::Embedding(
        torch::nn::EmbeddingOptions(FLAGS_vocab_size, FLAGS_embed_size));
    this->embedding = new_embedding;
    register_module("embedding", embedding);
  }

  torch::Tensor forward(torch::Tensor indices) {
    // std::cout << "in forward weight" << embedding->weight.sizes() << std::endl;
    // indices = indices.to(embedding->weight.device());
    // indices don't need gradient
    // indices.set_requires_grad(false);
    auto res = embedding->forward(indices);
    // std::cout << "in forward" << res.device() << std::endl;
    // std::cout << "in forward" << res.sizes() << std::endl;
    // std::cout << "in forward" << indices.sizes() << std::endl;

    return res;
  }
};

TORCH_MODULE(EmbeddingModelClonable);

template <typename DataLoader>
void train(int32_t epoch, EmbeddingModelClonable &model, torch::Device device,
           DataLoader &data_loader, torch::optim::Optimizer &optimizer) {
  model->train();
  for (auto &batch : data_loader) {
    xmh::Timer timer("OneStep");
    auto data = batch.data();
    optimizer.zero_grad();
    std::vector<torch::Device> devices;
    for (int i = 0; i < FLAGS_nr_gpus; i++) {
      devices.push_back(torch::Device(torch::kCUDA, i));
    }
    auto output =
        torch::nn::parallel::data_parallel(model, *data, devices, device);

	
    assert(output.device() == torch::Device(torch::kCUDA, 0));

    // std::cout << "Embedding output: " << output.sizes() << std::endl;
    // std::cout << "Embedding output: " << device << std::endl;

    // torch::Tensor target = torch::randn({batch_size, embed_size}).to(device);
    // torch::Tensor loss = torch::mse_loss(output, target).to(device);

    torch::Tensor loss = torch::sum(output).to(device);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();
    std::cout << "Loss: " << loss.item<float>() << std::endl;
    timer.end();
  }
}

int main(int argc, char **argv) {
  folly::init(&argc, &argv, true);
  xmh::Reporter::StartReportThread();


  int batch_size = FLAGS_batch_size;

  // torch::autograd::AnomalyMode::set_enabled(true);

  auto device = torch::cuda::is_available() ? torch::Device(torch::kCUDA, 0)
                                            : torch::kCPU;

  void *indices_buffers = nullptr;
  indices_buffers = malloc(
                    sizeof(int64_t) * batch_size * FLAGS_sample_number);
  
  for (int64_t i = 0; i < batch_size * FLAGS_sample_number; i++) {
    ((int64_t *)indices_buffers)[i] = 0;
  }
  torch::Tensor indices = torch::from_blob(
      indices_buffers, {FLAGS_sample_number, batch_size}, torch::kInt64);
  auto dataset = torch::data::datasets::TensorDataset(indices);
  auto train_loader =
      torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
          std::move(dataset),
          torch::data::DataLoaderOptions().batch_size(batch_size));

  EmbeddingModelClonable model(FLAGS_vocab_size, FLAGS_embed_size, device);
  model->to(device);
  for (const auto &p : model->parameters()) {
    std::cout << "model parameters" << p.device() << std::endl;
  }

  // 定义优化器
  torch::optim::SGD optimizer(model->parameters(),
                              torch::optim::SGDOptions(0.001));

	while(1){
    train(0, model, device, *train_loader, optimizer);
  }

  return 0;
}
