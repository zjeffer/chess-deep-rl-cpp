#pragma once


#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

struct ConvBlockImpl : public torch::nn::Module {
    ConvBlockImpl(int input_filters, int output_filters) {
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(input_filters, output_filters, 3).padding(1).stride(1)));
        batchNorm1 = register_module("batchNorm1", torch::nn::BatchNorm2d(output_filters));
    }

    torch::Tensor forward(const torch::Tensor &x) {
        return torch::relu(batchNorm1(conv1(x)));
    }

    torch::nn::Conv2d conv1 = nullptr;
    torch::nn::BatchNorm2d batchNorm1 = nullptr;
};
TORCH_MODULE(ConvBlock);
