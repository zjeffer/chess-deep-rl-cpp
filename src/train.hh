#pragma once

#include "neuralnet.hh"
#include "dataset.hh"
#include <memory>

class Trainer {
	public:
		Trainer(const std::string &networkPath, int batch_size, float learning_rate);
		~Trainer();

		void train();

	private:
		std::shared_ptr<NeuralNetwork> m_NN;
		int m_BatchSize;
		float m_LearningRate;

};