#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>

#include "neuralnet.hh"
#include "utils.hh"
#include "common.hh"

#define INPUT_CHANNELS 119
#define CONV_FILTERS 256
#define PLANE_SIZE 8
#define OUTPUT_PLANES 73
#define OUTPUT_SIZE (OUTPUT_PLANES * PLANE_SIZE * PLANE_SIZE)
#define POLICY_FILTERS 2
#define VALUE_FILTERS 1

bool NeuralNetwork::loadModel(std::string path) {
    try {
        // load model from path
        LOG(INFO) << "Loading model from: " << path;
        torch::load(this->neuralNet, path);
    } catch (const std::exception& e) {
        LOG(WARNING) << "Error loading model: " << e.what();
        return false;
    }
    return true;
}

bool NeuralNetwork::saveModel(std::string path, bool isTrained){
    try {
        if (path.size() == 0) {
            path = "./models/model_" + utils::getTimeString() + ".pt";
        }
        if (std::filesystem::exists(path)) {
            LOG(WARNING) << "Model already exists at: " << path << ". Overwriting...";
            std::filesystem::remove(path);
        }
        if (isTrained) {
            path.append("_trained");
        }
        // save model to path
        LOG(INFO) << "Saving model to: " << path;
        torch::save(this->neuralNet, path);
    } catch (const std::exception& e) {
        LOG(WARNING) << "Error saving model: " << e.what();
        return false;
    }
    return true;
}

NeuralNetwork::NeuralNetwork(std::string path, bool useCPU) {
    LOG(DEBUG) << "Creating NeuralNetwork object...";

    // set correct device
    if (useCPU) {
        LOG(DEBUG) << "Running on CPU.";
        this->device = torch::Device(torch::kCPU);
    } else {
        if (torch::cuda::is_available() ) {
            LOG(DEBUG) << "CUDA loaded. Device count: " << torch::cuda::device_count();
            this->device = torch::Device(torch::kCUDA);
        } else {
            LOG(WARNING) << "CUDA not available. Running on CPU.";
            this->device = torch::Device(torch::kCPU);
        }
    }

    this->neuralNet = Network(INPUT_CHANNELS, PLANE_SIZE, PLANE_SIZE, OUTPUT_SIZE, CONV_FILTERS, POLICY_FILTERS, VALUE_FILTERS);
    this->neuralNet->to(this->device);

    // if the path is given, load the model
    if (!path.empty()){
        // if path exists
        if (std::filesystem::is_regular_file(path)){
            this->loadModel(path);
        } else {
            LOG(WARNING) << "Model file at " << path << " does not exist. Creating new model.";
            if(!this->saveModel(path)){
                exit(EXIT_FAILURE);
            }
        }
    } else {
        LOG(WARNING) << "No model path given. Creating new model.";
        if(!this->saveModel("")){
            exit(EXIT_FAILURE);
        }
    }

    // random input
    LOG(DEBUG) << "Testing random input...";
    torch::Tensor input = torch::rand({1, 119, 8, 8});
    std::tuple<torch::Tensor, torch::Tensor> outputs = this->predict(input);
    LOG(DEBUG) << "Output successful";
}

Network NeuralNetwork::getNetwork() {
    return this->neuralNet;
}

std::tuple<torch::Tensor, torch::Tensor> NeuralNetwork::predict(torch::Tensor &input) {
    input = input.to(this->device);
    return this->neuralNet->forward(input);
}

void NeuralNetwork::trainBatches(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size) {
    float Loss = 0, Acc = 0;

    // enable training mode
    this->neuralNet->train();
    this->neuralNet->to(this->device);

	LOG(INFO) << "Starting training with " << data_size << " examples";
	int index = 0;
	for (auto batch : loader) {
		int size = batch.data.sizes()[0];
		LOG(INFO) << "Batch of size " << size;
		auto data = batch.data.to(torch::kCUDA);
		auto target = batch.target.to(torch::kCUDA);

        // check if policy_target contains nans
        if (!torch::nan_to_num(target).equal(target)){
            LOG(WARNING) << target;
            LOG(WARNING) << "Target contains nans";
            exit(EXIT_FAILURE);
        }

		// divide policy and value targets
		auto policy_target = target.slice(1, 0, 4672).view({size, 73, 8, 8});
		auto value_target = target.slice(1, 4672, 4673).view({size, 1});

        std::tuple<torch::Tensor, torch::Tensor> outputs = this->predict(data);
        torch::Tensor policy_output = std::get<0>(outputs);
        torch::Tensor value_output = std::get<1>(outputs);
        if (!torch::nan_to_num(policy_output).equal(policy_output)){
            LOG(WARNING) << policy_output;
            LOG(WARNING) << "Policy output contains nans";
            exit(EXIT_FAILURE);
        }
		policy_output = policy_output.view({size, 73, 8, 8});
		value_output = value_output.view({size, 1});

		// loss
		// policy loss is cross entropy loss
		auto policy_loss = -torch::sum(policy_target * torch::clamp(torch::log(policy_output), -10e1, 10e1));
		auto value_loss = torch::mse_loss(value_output, value_target);
		
		LOG(INFO) << "Policy loss: " << policy_loss;
		LOG(INFO) << "Value loss: " << value_loss;
		auto loss = torch::add(policy_loss, value_loss);

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();

		Loss += loss.item<float>();

		// calculate average loss
		auto end = std::min(data_size, (index + 1) * batch_size);
		LOG(INFO) << "===================== Epoch: " << index << " => Loss: " << Loss / (end);
		index += 1;
	}
	LOG(INFO) << "Training finished";

    // TODO: add timestamp
    this->saveModel("", true);
}