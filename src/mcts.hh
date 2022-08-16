#pragma once

#include "neuralnet.hh"
#include "node.hh"
#include "environment.hh"
#include "chess/thc.hh"
#include "tqdm.h"
#include "utils.hh"
#include "common.hh"

class MCTS {
  public:
    MCTS(Node* root, const std::shared_ptr<NeuralNetwork> &nn);
    ~MCTS();

    void run_simulations(int num_simulations);

    Node* select(Node* root);

    float expand(Node* node);

    void backpropagate(Node* node, float result);

    Node* getRoot();
    void setRoot(Node* newRoot);

    static int getTreeDepth(Node* root);

  private:
    Node* m_Root;
    std::shared_ptr<NeuralNetwork> m_NN;
};
