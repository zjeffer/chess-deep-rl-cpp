#pragma once

#include "mcts.hh"
#include "neuralnet.hh"

class Agent {
	public:
		Agent(std::string name, const std::shared_ptr<NeuralNetwork> &nn);

		MCTS* getMCTS();

		void updateMCTS(Node* newRoot);

		std::string getName();

		void setName(std::string name);

	private:
		std::shared_ptr<NeuralNetwork> nn;
		MCTS* mcts;
		std::string name;
};

