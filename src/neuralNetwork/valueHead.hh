#pragma once


#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

struct ValueHeadImpl : public torch::nn::Module {
    ValueHeadImpl(int inputFilters, int valueFilters, int width, int height, int linearNeurons) {
		const int valueArraySize = width * height * valueFilters;

        convValue = register_module("convValue", torch::nn::Conv2d(torch::nn::Conv2dOptions(inputFilters, valueFilters, 1).stride(1)));
        batchNormValue = register_module("batchNormValue", torch::nn::BatchNorm2d(valueFilters));

        linearValue1 = register_module("linearValue1", torch::nn::Linear(valueArraySize, linearNeurons));
		linearValue2 = register_module("linearValue2", torch::nn::Linear(linearNeurons, 1));
    }

    torch::Tensor forward(const torch::Tensor &input) {
		int size = input.size(0);

		auto value = convValue(input);
		value = value.view({size, -1});
		value = linearValue1(value);
		value = torch::relu(value);

		value = linearValue2(value);
		value = torch::tanh(value);

		return value;
	}


  private:
    torch::nn::Conv2d convValue = nullptr;
	torch::nn::BatchNorm2d batchNormValue = nullptr;
	torch::nn::Linear linearValue1 = nullptr;
	torch::nn::Linear linearValue2 = nullptr;
    
};
TORCH_MODULE(ValueHead);
