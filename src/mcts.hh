#ifndef mcts_hh
#define mcts_hh

#include "neuralnet.hh"
#include "node.hh"

class MCTS {
  public:
    MCTS(Node *root, NeuralNetwork *nn);
    ~MCTS();

    void run_simulations(int num_simulations);

    Node *select(Node *root);

    float expand(Node *node);

    void backpropagate(Node *node, float value);

    Node *getRoot();
    void setRoot(Node *newRoot);

    static int getTreeDepth(Node *root);

  private:
    Node *root;

    NeuralNetwork *nn;
};

#endif /* mcts_hh */