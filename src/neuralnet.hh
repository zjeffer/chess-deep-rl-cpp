#ifndef neuralnet_hh
#define neuralnet_hh

#include "environment.hh"

#include <array>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/api/include/torch/torch.h>
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


class NeuralNetwork : torch::nn::Module {
  public:
    NeuralNetwork();

    void buildNetwork();

    void predict(std::array<boolBoard, 19> &input,
                 std::array<floatBoard, 73> &output_probs, float &output_value);

    torch::Tensor forward(torch::Tensor x);

  private:
    torch::Device device = torch::Device(torch::kCUDA);

    torch::nn::Conv2d input_conv = nullptr, residual_conv = nullptr, policy_conv = nullptr, value_conv = nullptr;
    torch::nn::Linear policy_output = nullptr, lin2 = nullptr, lin3 = nullptr;
    torch::nn::BatchNorm2d bn1 = nullptr, bn2 = nullptr;

    torch::nn::Sequential build_policy_head();
    torch::nn::Sequential build_value_head();

};

#endif // neuralnet_hh