#pragma once

#include "dataset.hh"
#include "environment.hh"
#include "neuralNetwork/network.hh"

#include <array>
#include <vector>

using ChessDataLoader = torch::data::StatelessDataLoader<
	torch::data::datasets::MapDataset<
		ChessDataSet, 
		torch::data::transforms::Stack<
			torch::data::Example<
				at::Tensor, at::Tensor
			>
		>
	>, 
	torch::data::samplers::RandomSampler
>;

class NeuralNetwork {
  public:
    NeuralNetwork(std::string path = "", bool useCPU=false);
	~NeuralNetwork();

    std::pair<torch::Tensor, torch::Tensor> predict(torch::Tensor &input);

    void trainBatches(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size);

    bool loadModel(std::string path);

    bool saveModel(std::string path, bool isTrained = false);

	Network getNetwork();

  private:
    torch::Device device = torch::Device(torch::kCPU);

    Network m_NN = nullptr;

};

