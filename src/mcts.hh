#ifndef mcts_hh
#define mcts_hh

#include "node.hh"
#include "neuralnet.hh"

class MCTS {
   public:
    MCTS(Node* root);
    ~MCTS();

    void run_simulations(int num_simulations);

	Node* select(Node* root);

	float expand(Node* node);

	void backpropagate(Node* node, float value);

    Node* getRoot();

    static int getTreeDepth(Node* root);


   private:
    Node* root;

    NeuralNetwork nn;
};

#endif /* mcts_hh */