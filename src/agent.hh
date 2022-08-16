#pragma once

#include "mcts.hh"
#include "neuralnet.hh"

class Agent {
	public:
		Agent(std::string name, const std::shared_ptr<NeuralNetwork> &nn);

		MCTS* getMCTS();

		void updateMCTS(Node* newRoot);

		const std::string& getName() const;

		void setName(std::string name);

	private:
		std::shared_ptr<NeuralNetwork> m_NN;
		MCTS* m_MCTS;
		std::string m_Name;
};

