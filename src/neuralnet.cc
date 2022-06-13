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

#include <filesystem>

#include "utils.hh"

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

    input_conv->weight = torch::nn::init::xavier_uniform_(input_conv->weight);
    residual_conv->weight = torch::nn::init::xavier_uniform_(residual_conv->weight);
    policy_conv->weight = torch::nn::init::xavier_uniform_(policy_conv->weight);
    policy_output->weight = torch::nn::init::xavier_uniform_(policy_output->weight);
    value_conv->weight = torch::nn::init::xavier_uniform_(value_conv->weight);
    lin2->weight = torch::nn::init::xavier_uniform_(lin2->weight);
    lin3->weight = torch::nn::init::xavier_uniform_(lin3->weight);

    input_conv->bias = torch::nn::init::constant_(input_conv->bias, 0);
    residual_conv->bias = torch::nn::init::constant_(residual_conv->bias, 0);
    policy_conv->bias = torch::nn::init::constant_(policy_conv->bias, 0);
    policy_output->bias = torch::nn::init::constant_(policy_output->bias, 0);
    value_conv->bias = torch::nn::init::constant_(value_conv->bias, 0);
    lin2->bias = torch::nn::init::constant_(lin2->bias, 0);
    lin3->bias = torch::nn::init::constant_(lin3->bias, 0);

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

bool NeuralNetwork::loadModel(std::string path) {
    try {
        // load model from path
        G3LOG(INFO) << "Loading model from: " << path;
        torch::serialize::InputArchive ia;
        ia.load_from(path);
        this->load(ia);
    } catch (const std::exception& e) {
        G3LOG(WARNING) << "Error loading model: " << e.what();
        return false;
    }
    return true;
}

bool NeuralNetwork::saveModel(std::string path){
    try {
        if (path.size() == 0) {
            path = "./models/model_" + utils::getTimeString() + ".pt";
        }
        if (std::filesystem::exists(path)) {
            std::filesystem::remove(path);
        }
        // save model to path
        G3LOG(INFO) << "Saving model to: " << path;
        torch::serialize::OutputArchive oa;
        this->save(oa);
        oa.save_to(path);
    } catch (const std::exception& e) {
        G3LOG(WARNING) << "Error saving model: " << e.what();
        return false;
    }
    return true;
}

NeuralNetwork::NeuralNetwork(std::string path, bool useCPU) : torch::nn::Module() {
    G3LOG(DEBUG) << "Creating NeuralNetwork object...";

    if (useCPU) {
        G3LOG(DEBUG) << "Running on CPU.";
        this->device = torch::Device(torch::kCPU);
    } else {
        if (torch::cuda::is_available() ) {
            G3LOG(DEBUG) << "CUDA loaded. Device count: " << torch::cuda::device_count();
            this->device = torch::Device(torch::kCUDA);
        } else {
            G3LOG(WARNING) << "CUDA not available. Running on CPU.";
            this->device = torch::Device(torch::kCPU);
        }
    }

    this->buildNetwork();
    // if the path is given, load the model
    if (!path.empty()){
        // if path exists
        if (std::filesystem::is_regular_file(path)){
            this->loadModel(path);
        } else {
            G3LOG(WARNING) << "Model file does not exist. Creating new model.";
            if(!this->saveModel(path)){
                exit(EXIT_FAILURE);
            }
        }
    }

    // random input
    G3LOG(DEBUG) << "Testing random input...";
    torch::Tensor input = torch::rand({1, 119, 8, 8});
    torch::Tensor output;
    this->predict(input, output);
    if (output.dim() != 2){
        G3LOG(FATAL) << "Failed to test model with random input";
        exit(EXIT_FAILURE);
    }
    G3LOG(DEBUG) << "Output successful";
}

void NeuralNetwork::predict(torch::Tensor &input, torch::Tensor &output) {
    output = this->forward(input);
}

void NeuralNetwork::train(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size) {
    float Loss = 0, Acc = 0;

	G3LOG(INFO) << "Starting training with " << data_size << " examples";
	int index = 0;
	for (auto batch : loader) {
		int size = batch.data.sizes()[0];
		G3LOG(INFO) << "Batch of size " << size;
		auto data = batch.data.to(torch::kCUDA);
		auto target = batch.target.to(torch::kCUDA);

        // check if policy_target contains nans
        if (!torch::nan_to_num(target).equal(target)){
            G3LOG(WARNING) << target;
            G3LOG(WARNING) << "Target contains nans";
            exit(EXIT_FAILURE);
        }

		// divide policy and value targets
		auto policy_target = target.slice(1, 0, 4672).view({size, 73, 8, 8});
		auto value_target = target.slice(1, 4672, 4673).view({size, 1});

		auto output = this->forward(data);
        if (!torch::nan_to_num(output).equal(output)){
            G3LOG(WARNING) << output;
            G3LOG(WARNING) << "Policy output contains nans";
            exit(EXIT_FAILURE);
        }
		auto policy_output = output.slice(1, 0, 4672).view({size, 73, 8, 8});
		auto value_output = output.slice(1, 4672, 4673).view({size, 1});

		// loss
		// policy loss is cross entropy loss
		auto policy_loss = -torch::sum(policy_target * torch::clamp(torch::log(policy_output), -10e1, 10e1));
		auto value_loss = torch::mse_loss(value_output, value_target);
		
		G3LOG(INFO) << "Policy loss: " << policy_loss;
		G3LOG(INFO) << "Value loss: " << value_loss;
		auto loss = torch::add(policy_loss, value_loss);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.item<float>();

		// calculate average loss
		auto end = std::min(data_size, (index + 1) * batch_size);
		G3LOG(INFO) << "===================== Epoch: " << index << " => Loss: " << Loss / (end);
		index += 1;
	}
	G3LOG(INFO) << "Training finished";

    // TODO: add timestamp
    this->saveModel("models/model.pt");
}