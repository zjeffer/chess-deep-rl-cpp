#ifndef NEURALNET_HH
#define NEURALNET_HH

#include "environment.hh"
#include "dataset.hh"

#include <ATen/core/TensorBody.h>
#include <array>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/data/dataloader/stateful.h>
#include <torch/nn/modules/conv.h>
#include <vector>

#include <torch/nn/module.h>
#include <torch/nn/modules/activation.h>
#include <torch/nn/modules/batchnorm.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/linear.h>
#include <torch/nn/options/batchnorm.h>
#include <torch/nn/options/conv.h>
#include <torch/nn/options/linear.h>


class NeuralNetwork : public torch::nn::Module {
  public:
    NeuralNetwork();

    void buildNetwork();

    void predict(std::array<boolBoard, 19> &input,
                 std::array<floatBoard, 73> &output_probs, float &output_value);

    torch::Tensor forward(torch::Tensor x);

    void train(torch::data::StatelessDataLoader<ChessDataSet, torch::data::samplers::SequentialSampler> &loader, torch::optim::Optimizer& optimizer, int data_size);

  private:
    // set the device depending on if cuda is available
    torch::Device device = torch::Device(torch::kCPU);

    torch::nn::Conv2d input_conv = nullptr, residual_conv = nullptr, policy_conv = nullptr, value_conv = nullptr;
    torch::nn::Linear policy_output = nullptr, lin2 = nullptr, lin3 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr, bn2 = nullptr;

    torch::nn::Sequential build_policy_head();
    torch::nn::Sequential build_value_head();

};

#endif // NEURALNET_HH