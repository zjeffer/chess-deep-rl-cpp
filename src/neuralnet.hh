#ifndef neuralnet_hh
#define neuralnet_hh

#include <array>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/nn/modules/conv.h>
#include <vector>

#include "board.hh"
#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/linear.h>

struct Net : torch::nn::Module {
    Net() {
        conv1 = register_module("input_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(19, 256, 3).padding(1).stride(1)));
        // conv layer for residual blocks
        conv2 = register_module("residual_layer", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1).stride(1)));

        // policy head
        conv3 = register_module("policy_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 2, 1).padding(1).stride(1)));
        lin1 = register_module("policy_output", torch::nn::Linear(2, 8 * 8 * 73));

        // // value head
        conv4 = register_module("value_conv", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 1, 1).padding(1).stride(1)));
        lin2 = register_module("value_lin", torch::nn::Linear(1, 256));
        lin3 = register_module("value_output", torch::nn::Linear(256, 1));
    }

    torch::nn::Sequential policy_head() {
        return torch::nn::Sequential(
            conv3,
            torch::nn::BatchNorm2d(2),
            torch::nn::ReLU(),
            torch::nn::Flatten(),
            lin1,
            torch::nn::ReLU());
    }

    torch::nn::Sequential value_head() {
        return torch::nn::Sequential(
            conv4,
            torch::nn::BatchNorm2d(1),
            torch::nn::ReLU(),
            torch::nn::Flatten(),
            lin2, 
			torch::nn::ReLU(),
            lin3,
            torch::nn::Sigmoid());
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(conv1->forward(x));

        // add residual blocks
        for (int i = 0; i < 19; i++) {
            // skip connection
            torch::Tensor skip_connection_input = x.detach().clone();
            x = torch::relu(conv2->forward(x));
            x = torch::add(conv2->forward(x), skip_connection_input);
            x = torch::relu(x);
        }

        // add policy head and value head
        torch::Tensor policy_output = policy_head()->forward(x);
        torch::Tensor value_output = value_head()->forward(x);

        return torch::cat({policy_output, value_output}, 1);
    }

    torch::nn::Conv2d conv1 = nullptr, conv2 = nullptr, conv3 = nullptr, conv4 = nullptr;
    torch::nn::Linear lin1 = nullptr, lin2 = nullptr, lin3 = nullptr;
};

class NeuralNetwork {
  public:
    NeuralNetwork();

    void buildNetwork();

    void predict(std::array<boolBoard, 19> &input,
                 std::array<floatBoard, 73> &output_probs, float &output_value);

  private:
    struct Net model;

    torch::Device device = torch::Device(torch::kCPU);
};

#endif // neuralnet_hh