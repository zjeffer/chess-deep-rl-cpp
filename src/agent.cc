#include "agent.hh"

Agent::Agent(std::string name, const std::shared_ptr<NeuralNetwork> &nn)
    : m_NN(nn), m_MCTS(new MCTS(new Node(), m_NN)), m_Name(name) {

}


MCTS* Agent::getMCTS() {
    return m_MCTS;
}

void Agent::updateMCTS(Node* newRoot){
    m_MCTS->setRoot(newRoot);
}

const std::string& Agent::getName() const {
    return m_Name;
}

void Agent::setName(std::string name) {
    m_Name = name;
}