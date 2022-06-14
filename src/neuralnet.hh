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

#define CONV_FILTERS 256
#define PLANE_SIZE 8
#define OUTPUT_PLANES 73
#define OUTPUT_SIZE (OUTPUT_PLANES * PLANE_SIZE * PLANE_SIZE)
#define POLICY_FILTERS 2
#define VALUE_FILTERS 1

struct ConvBlock : torch::nn::Module {
	ConvBlock(int in_filters = CONV_FILTERS) {
		register_module("conv1", conv1 = torch::nn::Conv2d(torch::nn::Conv2dOptions(in_filters, CONV_FILTERS, 3).stride(1).padding(1)));
		register_module("batchNorm1", batchNorm1 = torch::nn::BatchNorm2d(CONV_FILTERS));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = conv1->forward(x);
		x = batchNorm1->forward(x);
		x = torch::relu(x);
		return x;
	}

	torch::nn::Conv2d conv1 = nullptr;
	torch::nn::BatchNorm2d batchNorm1 = nullptr;
};

struct ResidualBlock : torch::nn::Module {
	ResidualBlock() {
		// TODO: register modules?
	}

	torch::Tensor forward(torch::Tensor x) {
		x = convBlock1.forward(x);
		x = batchNorm1->forward(x);
		x = torch::relu(x);
		x = convBlock2.forward(x);
		x = batchNorm2->forward(x);
		return x;
	}

	torch::nn::BatchNorm2d batchNorm1 = nullptr, batchNorm2 = nullptr;
	ConvBlock convBlock1, convBlock2;
};

struct Net : public torch::nn::Module {
	Net() {
		// input conv block, then 19 residual blocks
		
	}

	torch::nn::Conv2d input_conv = nullptr;
};




class NeuralNetwork : public torch::nn::Module {
  public:
    NeuralNetwork(std::string path = "", bool useCPU=false);

    void buildNetwork();

    void predict(torch::Tensor &input, torch::Tensor &output);

    torch::Tensor forward(torch::Tensor x);

    void trainBatches(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size);

    bool loadModel(std::string path);

    bool saveModel(std::string path, bool isTrained = false);

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