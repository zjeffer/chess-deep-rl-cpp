#pragma once

#include "neuralnet.hh"
#include "dataset.hh"

class Trainer {
	public:
		Trainer(const std::string& networkPath);
		~Trainer();

		void train();

	private:
		NeuralNetwork* nn;

};