#include "agent.hh"

Agent::Agent() {
    this->nn = new NeuralNetwork();
    this->mcts = new MCTS(new Node(), nn);
}

Agent::Agent(std::string name): Agent() {
    this->name = name;
}

MCTS* Agent::getMCTS() {
    return this->mcts;
}

void Agent::updateMCTS(Node* newRoot){
    this->mcts->setRoot(newRoot);
}