#pragma once


#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

struct PolicyHeadImpl : public torch::nn::Module {
    PolicyHeadImpl(int inputFilters, int policyFilters, int width, int height, int outputs) {
        int policyArraySize = width * height * policyFilters; // 8 x 8 x 2 = 128

        convPolicy = register_module("convPolicy", torch::nn::Conv2d(torch::nn::Conv2dOptions(inputFilters, policyFilters, 1).stride(1)));
        batchNormPolicy = register_module("batchNormPolicy", torch::nn::BatchNorm2d(policyFilters));
        linearPolicy = register_module("linearPolicy", torch::nn::Linear(policyArraySize, outputs));
    }

    torch::Tensor forward(const torch::Tensor &input) {
        int size = input.size(0);

        // conv block
        auto pol = convPolicy(input);
        pol = batchNormPolicy(pol);
        pol = torch::relu(pol);
        
        // flatten
        pol = pol.view({size, -1});
        
        // linear
        pol = linearPolicy(pol);

        // softmax activation
        pol = torch::softmax(pol, 1);
        return pol;
	}


  private:
    torch::nn::Conv2d convPolicy = nullptr;
	torch::nn::BatchNorm2d batchNormPolicy = nullptr;
	torch::nn::Linear linearPolicy = nullptr;
    
};
TORCH_MODULE(PolicyHead);
