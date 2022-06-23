#ifndef NEURALNET_HH
#define NEURALNET_HH

#include "dataset.hh"
#include "environment.hh"
#include "neuralNetwork/network.hh"

#include <array>
#include <torch/torch.h>
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

    std::tuple<torch::Tensor, torch::Tensor> predict(torch::Tensor &input);

    void trainBatches(ChessDataLoader &loader, torch::optim::Optimizer &optimizer, int data_size, int batch_size);

    bool loadModel(std::string path);

    bool saveModel(std::string path, bool isTrained = false);

	Network getNetwork();

  private:
    torch::Device device = torch::Device(torch::kCPU);

    Network neuralNet = nullptr;

};

#endif // NEURALNET_HH