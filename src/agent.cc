#include "agent.hh"

Agent::Agent(std::string name, NeuralNetwork* nn) {
    this->nn = nn;
    this->mcts = new MCTS(new Node(), nn);
}


MCTS* Agent::getMCTS() {
    return this->mcts;
}

void Agent::updateMCTS(Node* newRoot){
    this->mcts->setRoot(newRoot);
}