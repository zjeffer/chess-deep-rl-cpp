#ifndef AGENT_HH
#define AGENT_HH

#include "mcts.hh"
#include "neuralnet.hh"

class Agent {
	public:
		Agent();
		Agent(std::string name);

		MCTS* getMCTS();

		void updateMCTS(Node* newRoot);

		std::string name;

	private:
		NeuralNetwork* nn;
		MCTS* mcts;
};

#endif // AGENT_HH