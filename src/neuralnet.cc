#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <filesystem>

#include "neuralnet.hh"
#include "utils.hh"
#include "types.hh"
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
        torch::load(m_NN, path);
    } catch (const std::exception& e) {
        LOG(WARNING) << "Error loading model: " << e.what();
        return false;
    }
    return true;
}

bool NeuralNetwork::saveModel(std::string path, bool isTrained){
    try {
        if (path.size() == 0) {
            path = "./models/model_" + utils::getTimeString();
        }
        if (isTrained) {
            path.append("_trained");
        }
        if (!path.ends_with(".pt")) {
            path.append(".pt");
        }
        if (std::filesystem::exists(path)) {
            LOG(WARNING) << "Model already exists at: " << path << ". Overwriting...";
            std::filesystem::remove(path);
        }
        // save model to path
        LOG(INFO) << "Saving model to: " << path;
        torch::save(m_NN, path);
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

    m_NN = Network(INPUT_CHANNELS, PLANE_SIZE, PLANE_SIZE, OUTPUT_SIZE, CONV_FILTERS, POLICY_FILTERS, VALUE_FILTERS);
    m_NN->eval();
    m_NN->to(this->device);

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
    std::pair<torch::Tensor, torch::Tensor> outputs = this->predict(input);
    LOG(DEBUG) << "Output successful";
}

NeuralNetwork::~NeuralNetwork() {
    std::cout << "Deleting NeuralNetwork object..." << std::endl;
}

Network NeuralNetwork::getNetwork() {
    return m_NN;
}

std::pair<torch::Tensor, torch::Tensor> NeuralNetwork::predict(torch::Tensor &input) {
    input = input.to(this->device);
    return m_NN->forward(input);
}

std::tuple<torch::Tensor, torch::Tensor> loss_function(std::tuple<torch::Tensor, torch::Tensor> outputs, torch::Tensor target) {
    torch::Tensor policy_output = std::get<0>(outputs);
    torch::Tensor value_output = std::get<1>(outputs);

    // divide policy and value targets
    int size = policy_output.sizes()[0];
    auto policy_target = target.slice(1, 0, 4672).view({size, 73, 8, 8});
    auto value_target = target.slice(1, 4672, 4673).view({size, 1});
    
    if (!torch::nan_to_num(policy_output).equal(policy_output)){
        LOG(WARNING) << policy_output;
        LOG(WARNING) << "Policy output contains nans";
        exit(EXIT_FAILURE);
    }

    policy_output = policy_output.view({size, 73, 8, 8});
    value_output = value_output.view({size, 1});

    // loss
    torch::Tensor policy_loss = -torch::log_softmax(policy_output, 1).mul(policy_target).sum(1).mean();
    torch::Tensor value_loss = torch::mse_loss(value_output, value_target);

    return std::make_tuple(policy_loss, value_loss);
}

void NeuralNetwork::trainBatches(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size) {
    float Loss = 0;

    // enable training mode
    m_NN->train();
    m_NN->to(this->device);

    LossHistory loss_history;
    loss_history.batch_size = batch_size;
    loss_history.data_size = data_size;

	LOG(INFO) << "Starting training with " << data_size << " examples";
	int index = 0;
	for (auto batch : loader) {
        if (!g_Running) {
            exit(EXIT_FAILURE);
        }

        optimizer.zero_grad();
		int size = batch.data.sizes()[0];
		auto input = batch.data.to(this->device);
		auto target = batch.target.to(this->device);

        // check if policy_target contains nans
        if (!torch::nan_to_num(target).equal(target)){
            LOG(WARNING) << target;
            LOG(WARNING) << "Target contains nans";
            exit(EXIT_FAILURE);
        }
	
        std::tuple<torch::Tensor, torch::Tensor> outputs = this->predict(input);
		std::tuple<torch::Tensor, torch::Tensor> losses = loss_function(outputs, target);

        torch::Tensor policy_loss = std::get<0>(losses);
        torch::Tensor value_loss = std::get<1>(losses);
        torch::Tensor loss = policy_loss + value_loss;
		
		loss.backward();
		optimizer.step();


		// calculate average loss
		Loss += loss.item<float>();
		auto end = std::min(data_size, (index + 1) * batch_size);
		LOG(INFO) << "======== Epoch: " << index << ". Batch size: " << size << " => Loss: " << Loss / (end)
            << ". Policy loss: " << policy_loss.item<float>() << ". Value loss: " << value_loss.item<float>() << " ========";

        // add loss to history
        loss_history.losses.push_back(loss.item<float>());
        loss_history.policies.push_back(policy_loss.item<float>());
        loss_history.values.push_back(value_loss.item<float>());
        loss_history.historySize++;

		index++;
	}
	LOG(INFO) << "Training finished";

    std::string timeString = utils::getTimeString();
    // save loss history to csv, to make graphs with
    utils::writeLossToCSV("losses/history_" + timeString + ".csv", loss_history);
    // save the trained model
    this->saveModel("./models/model_" + timeString, true);

    // set network back to evaluation mode
    m_NN->eval();
}