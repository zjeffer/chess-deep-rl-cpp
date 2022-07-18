#include "agent.hh"

Agent::Agent(std::string name, const std::shared_ptr<NeuralNetwork> &nn) {
    this->name = name;
    this->nn = nn;
    this->mcts = new MCTS(new Node(), nn);
}


MCTS* Agent::getMCTS() {
    return this->mcts;
}

void Agent::updateMCTS(Node* newRoot){
    this->mcts->setRoot(newRoot);
}

std::string Agent::getName() {
    return this->name;
}

void Agent::setName(std::string name) {
    this->name = name;
}