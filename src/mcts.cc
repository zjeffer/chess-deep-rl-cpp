#include "mcts.hh"
#include "environment.hh"
#include "neuralnet.hh"
#include "tqdm.h"
#include <stdio.h>

#include "chess/thc.hh"
#include <chrono>
#include <iostream>



MCTS::MCTS(Node* root, NeuralNetwork* nn) {
	this->root = root;
	this->nn = nn;
}

MCTS::~MCTS() {
	delete this->root;
}


void MCTS::run_simulations(int num_simulations) {
	printf("Running %d simulations...\n", num_simulations);
	tqdm bar;
	for (int i = 0; i < num_simulations; i++) {
		bar.progress(i, num_simulations);
		// selection
		Node* leaf = select(this->root);

		// expansion and evaluation
		float value = expand(leaf);

		// backpropagation
		backpropagate(leaf, value);
	}
	// printf("Tree depth: %d\n", getTreeDepth(this->root));
}

Node* MCTS::select(Node* root){
	auto start_time = std::chrono::high_resolution_clock::now();

	Node* current = root;
	int traversals = 0;
    while (!current->isLeaf()) {
		traversals++;
        // get the max Q+U value
		std::vector<Node*> children = current->getChildren();
		// start with random child as best child
		if (children.size() == 0) {
			perror("No children");
			exit(EXIT_FAILURE);
		}
		Node* best_child = children[rand() % children.size()];
		float best_score = -1;
		for (int i = 0; i < (int)children.size(); i++) {
			std::cout << "Child " << i << ": " << children[i]->getQ() << " + " << children[i]->getUCB() << std::endl;
			Node* child = children[i];
			float score = child->getPUCTScore();
			if (score > best_score) {
				best_child = child;
				best_score = score;
			}
		}
		if (best_child == nullptr) {
			perror("Error: best_child is null");
			exit(EXIT_FAILURE);
		}
		current = best_child;
    }
	auto stop = std::chrono::high_resolution_clock::now();
	// std::cout << "Selection: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start_time).count() << " microseconds for " << traversals << " traversals" << std::endl;

	return current;
}

float MCTS::expand(Node* node){
	// convert board to input state
	Environment board = Environment(node->getFen());
	std::array<boolBoard, 19> inputState = board.boardToInput();
	
	// outputs
	std::array<floatBoard, 73> output_probs;
	float output_value = 0.0;

	// send input to neural network
	this->nn->predict(inputState, output_probs, output_value);

	// output to moves
	std::vector<thc::Move> legal_moves;
	board.getLegalMoves(legal_moves);
	
	std::map<thc::Move, float> moveProbs = board.outputProbsToMoves(output_probs, legal_moves);

	// add nodes for every move
	for (int i = 0; i < (int)legal_moves.size(); i++) {
		thc::Move move = legal_moves[i];
		// make the move on the board
		std::string new_fen = board.makeMove(move);

		// get the move probability for this move
		float prior = moveProbs[move];
		
		// create a new node
		Node* child = new Node(new_fen, node, move, prior);
		// add the new node to the children of the current node
		node->add_child(child);

		// undo move
		board.undoMove(move);
	}
	return output_value;
}


void MCTS::backpropagate(Node* node, float value){
	while (node != nullptr) {
		// printf("Backpropagating...\n");
		node->incrementVisit();
		node->setValue(value);
		node = node->getParent();
	}
}


Node* MCTS::getRoot() {
	return this->root;
}

void MCTS::setRoot(Node *newRoot){
	delete this->root;
	this->root = newRoot;
}

int MCTS::getTreeDepth(Node* root) {
	// recursive function of getting height of the tree
	if (root->isLeaf()) {
		return 0;
	}
	int max_depth = 0;
	for (int i = 0; i < (int)root->getChildren().size(); i++) {
		int depth = MCTS::getTreeDepth(root->getChildren()[i]);
		if (depth > max_depth) {
			max_depth = depth;
		}
	}
	return max_depth + 1;
}