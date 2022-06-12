#ifndef NEURALNET_HH
#define NEURALNET_HH

#include "dataset.hh"
#include "environment.hh"

#include <ATen/core/TensorBody.h>
#include <array>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/data/dataloader/stateful.h>
#include <torch/nn/modules/conv.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <vector>

#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/linear.h>

using ChessDataLoader = torch::data::StatelessDataLoader<
	torch::data::datasets::MapDataset<
		ChessDataSet, 
		torch::data::transforms::Stack<
			torch::data::Example<
				at::Tensor, at::Tensor
			>
		>
	>, 
	// torch::data::samplers::RandomSampler
	torch::data::samplers::SequentialSampler

>;

class NeuralNetwork : public torch::nn::Module {
  public:
    NeuralNetwork(std::string path = "", bool useCPU=false);

    void buildNetwork();

    void predict(torch::Tensor &input, torch::Tensor &output);

    torch::Tensor forward(torch::Tensor x);

    void train(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size);

    bool loadModel(std::string path);

    bool saveModel(std::string path);

  private:
    // set the device depending on if cuda is available
    torch::Device device = torch::Device(torch::kCPU);

    torch::nn::Conv2d input_conv = nullptr, residual_conv = nullptr, policy_conv = nullptr, value_conv = nullptr;
    torch::nn::Linear policy_output = nullptr, lin2 = nullptr, lin3 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr, bn2 = nullptr;

    void build_policy_head();
    void build_value_head();

    torch::nn::Sequential policy_head = nullptr, value_head = nullptr;
};

#endif // NEURALNET_HH