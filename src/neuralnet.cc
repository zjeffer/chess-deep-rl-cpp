#include "neuralnet.hh"
#include <ATen/Functions.h>
#include <ATen/core/TensorBody.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceType.h>
#include <c10/core/GradMode.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <torch/data.h>
#include <torch/data/dataloader.h>
#include <torch/data/datasets/base.h>
#include <torch/nn/init.h>
#include <torch/nn/module.h>
#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/modules/conv.h>
#include <torch/nn/pimpl.h>

#define CONV_FILTERS 256
#define PLANE_SIZE 8
#define OUTPUT_PLANES 73
#define OUTPUT_SIZE (OUTPUT_PLANES * PLANE_SIZE * PLANE_SIZE)
#define POLICY_FILTERS 2
#define VALUE_FILTERS 1

void NeuralNetwork::buildNetwork() {
    input_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(119, CONV_FILTERS, 3).padding(1).stride(1));
    residual_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(CONV_FILTERS, CONV_FILTERS, 3).padding(1).stride(1));
    policy_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(CONV_FILTERS, POLICY_FILTERS, 1).stride(1));
    policy_output = torch::nn::Linear(POLICY_FILTERS * PLANE_SIZE * PLANE_SIZE, OUTPUT_SIZE);
    value_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(CONV_FILTERS, VALUE_FILTERS, 1).stride(1));
    lin2 = torch::nn::Linear(VALUE_FILTERS * 8 * 8, 256);
    lin3 = torch::nn::Linear(256, 1);

    torch::nn::init::xavier_uniform_(input_conv->weight);
    torch::nn::init::xavier_uniform_(residual_conv->weight);
    torch::nn::init::xavier_uniform_(policy_conv->weight);
    torch::nn::init::xavier_uniform_(policy_output->weight);
    torch::nn::init::xavier_uniform_(value_conv->weight);
    torch::nn::init::xavier_uniform_(lin2->weight);
    torch::nn::init::xavier_uniform_(lin3->weight);

    torch::nn::init::constant_(input_conv->bias, 0);
    torch::nn::init::constant_(residual_conv->bias, 0);
    torch::nn::init::constant_(policy_conv->bias, 0);
    torch::nn::init::constant_(policy_output->bias, 0);
    torch::nn::init::constant_(value_conv->bias, 0);
    torch::nn::init::constant_(lin2->bias, 0);
    torch::nn::init::constant_(lin3->bias, 0);

    // main input
    input_conv = register_module("input_conv", input_conv);

    // conv layer for residual blocks
    residual_conv = register_module("residual_layer", residual_conv);

    // policy head
    policy_conv = register_module("policy_conv", policy_conv);
    policy_output = register_module("policy_output", policy_output);

    // // value head
    value_conv = register_module("value_conv", value_conv);
    lin2 = register_module("value_lin", lin2);
    lin3 = register_module("value_output", lin3);

    // all layers to device
    input_conv->to(device);
    residual_conv->to(device);
    policy_conv->to(device);
    policy_output->to(device);
    value_conv->to(device);
    lin2->to(device);
    lin3->to(device);

    // move this to device
    this->to(device);

    this->build_policy_head();
    this->build_value_head();
}

void NeuralNetwork::build_policy_head() {
    this->policy_head = torch::nn::Sequential(
        policy_conv,
        torch::nn::BatchNorm2d(POLICY_FILTERS),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        policy_output,
        torch::nn::ReLU());
    this->policy_head->to(device);
}

void NeuralNetwork::build_value_head() {
        this->value_head = torch::nn::Sequential(
        value_conv,
        torch::nn::BatchNorm2d(VALUE_FILTERS),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        lin2,
        torch::nn::ReLU(),
        lin3,
        torch::nn::Tanh());
    this->value_head->to(this->device);

}

torch::Tensor NeuralNetwork::forward(torch::Tensor x) {
    x = x.to(this->device);
    x = torch::relu(input_conv->forward(x));

    // add residual blocks
    for (int i = 0; i < 19; i++) {
        // skip connection
        torch::Tensor skip_connection_input = x.detach().clone();
        x = torch::relu(residual_conv->forward(x));
        x = torch::add(residual_conv->forward(x), skip_connection_input);
        x = torch::relu(x);
    }

    return torch::cat({this->policy_head->forward(x), this->value_head->forward(x)}, 1);
}

NeuralNetwork::NeuralNetwork(bool useCPU) {
    std::cout << "Creating NeuralNetwork object..." << std::endl;

    if (useCPU) {
        std::cout << "Running on CPU." << std::endl;
        this->device = torch::Device(torch::kCPU);
    } else {
        if (torch::cuda::is_available() ) {
            std::cout << "CUDA loaded. Device count: " << torch::cuda::device_count() << std::endl;
            this->device = torch::Device(torch::kCUDA);
        } else {
            std::cout << "CUDA not available. Running on CPU." << std::endl;
            this->device = torch::Device(torch::kCPU);
        }
    }

    this->buildNetwork();

    // random input
    std::cout << "Testing random input..." << std::endl;
    torch::Tensor input = torch::rand({1, 119, 8, 8});
    torch::Tensor output = this->forward(input);
    if (output.dim() != 2){
        std::cerr << "Failed to test model with random input" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Output successful" << std::endl;
}

void NeuralNetwork::predict(torch::Tensor &input, torch::Tensor &output) {
    output = this->forward(input);
}

template <typename DataLoader>
void NeuralNetwork::train(DataLoader &loader, torch::optim::Optimizer &optimizer, int data_size) {
    int index = 0;
    float Loss = 0.0, Acc = 0.0;

    for (auto &batch : loader) {
        auto data = batch.data.to(this->device);
        // TODO: what does this function do? what is view?
        auto targets = batch.target.to(this->device).view({-1});

        auto output = this->forward(data);
        // probs_output = the first 4672 floats
        auto probs_output = output.slice(1, 0, 4672);
        // value_output = the last float
        auto value_output = output.slice(1, 4672, 4672 + 1);
        auto probs_loss = torch::cross_entropy_loss(probs_output, targets);
        auto value_loss = torch::mse_loss(value_output, targets);

        auto loss = probs_loss + value_loss;
        auto acc = output.argmax(1).eq(targets).sum();

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        Loss += loss.template item<float>();
        Acc += acc.template item<float>();

        if (index++ % 2 == 0) {
            auto end = std::min(data_size, (index + 1) * 8);

            std::cout << "Train" << "\tLoss: " << Loss / end << "\tAcc: " << Acc / end << std::endl;
        }
        // TODO (https://github.com/pytorch/examples/blob/master/cpp/custom-dataset/custom-dataset.cpp)
    }

    // TODO: add date & time
    torch::save(this, "./models/model.pt");
}