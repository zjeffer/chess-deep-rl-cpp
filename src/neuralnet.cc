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
    input_conv = torch::nn::Conv2d(torch::nn::Conv2dOptions(19, CONV_FILTERS, 3).padding(1).stride(1));
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
    torch::Device device = torch::Device(torch::kCUDA);
    input_conv->to(device);
    residual_conv->to(device);
    policy_conv->to(device);
    policy_output->to(device);
    value_conv->to(device);
    lin2->to(device);
    lin3->to(device);

    // move this to device
    this->to(device);
}

torch::nn::Sequential NeuralNetwork::build_policy_head() {
    torch::nn::Sequential policy_head = torch::nn::Sequential(
        policy_conv,
        torch::nn::BatchNorm2d(POLICY_FILTERS),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        policy_output,
        torch::nn::ReLU());
    policy_head->to(device);
    return policy_head;
}

torch::nn::Sequential NeuralNetwork::build_value_head() {
    torch::nn::Sequential value_head = torch::nn::Sequential(
        value_conv,
        torch::nn::BatchNorm2d(VALUE_FILTERS),
        torch::nn::ReLU(),
        torch::nn::Flatten(),
        lin2,
        torch::nn::ReLU(),
        lin3,
        torch::nn::Tanh());
    value_head->to(this->device);
    return value_head;
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

    // add policy head and value head
    torch::Tensor policy_output = this->build_policy_head()->forward(x);
    torch::Tensor value_output = this->build_value_head()->forward(x);

    return torch::cat({policy_output, value_output}, 1);
}

NeuralNetwork::NeuralNetwork() {
    std::cout << "Creating NeuralNetwork object..." << std::endl;

    if (torch::cuda::is_available()) {
        std::cout << "CUDA loaded." << std::endl;
        std::cout << "CUDA device count: " << torch::cuda::device_count() << std::endl;
        std::cout << "CUDNN: " << torch::cuda::cudnn_is_available() << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not initialized. Using CPU." << std::endl;
        device = torch::Device(torch::kCPU);
    }

    this->buildNetwork();

    // random input
    torch::Tensor input = torch::rand({1, 19, 8, 8});
    torch::Tensor output = this->forward(input);
}

void NeuralNetwork::predict(std::array<boolBoard, 19> &input, std::array<floatBoard, 73> &output_probs, float &output_value) {
    // arrays to tensors
    torch::Tensor input_tensor = torch::from_blob(input.data(), {1, 19, 8, 8});

    torch::Tensor output_tensor = this->forward(input_tensor);
    // send tensor to cpu so we can access it
    output_tensor = output_tensor.to(torch::Device(torch::kCPU));

    // TODO: fix nans?
    output_tensor = output_tensor.nan_to_num();

    // tensor to array
    float *output_probs_array = output_tensor.data_ptr<float>();

    // reshape output_probs_array to output_probs (4672 floats to 73 floatboards)
    for (int i = 0; i < 73; i++) {
        for (int j = 0; j < 8; j++) {
            for (int k = 0; k < 8; k++) {
                output_probs[i].board[j][k] = output_probs_array[i * 8 * 8 + j * 8 + k];
            }
        }
    }

    output_value = output_probs_array[4672];

    // std::cout << "output_probs example: " << output_probs[0].board[0][0] << std::endl;
    // std::cout << "output_value example: " << output_value << std::endl;
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

    torch::save(this, "./models/model.pt");
}