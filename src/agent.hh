#pragma once

#include "mcts.hh"
#include "neuralnet.hh"

class Agent {
	public:
		Agent(std::string name = "", NeuralNetwork* nn = new NeuralNetwork());

		MCTS* getMCTS();

		void updateMCTS(Node* newRoot);

		std::string getName();

		void setName(std::string name);

	private:
		NeuralNetwork* nn;
		MCTS* mcts;
		std::string name;
};

