#include "mcts.hh"
#include "board.hh"
#include <stdio.h>

#include "chess/thc.hh"
#include <chrono>
#include <iostream>



MCTS::MCTS(Node* root) {
	this->root = root;
	this->nn = NeuralNetwork();
}

MCTS::~MCTS() {
	// delete this->root;
}


void MCTS::run_simulations(int num_simulations) {
	printf("Running %d simulations...\n", num_simulations);
	for (int i = 0; i < num_simulations; i++) {
		auto start_time = std::chrono::high_resolution_clock::now();
		// selection
		Node* leaf = select(this->root);

		// expansion and evaluation
		float value = expand(leaf);

		// backpropagation
		backpropagate(leaf, value);
		auto stop = std::chrono::high_resolution_clock::now();
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start_time);
		printf("Simulation %d took %li microseconds\n", i, duration.count());
		if (i % 500 == 0) {
			printf("Tree depth: %d\n", getTreeDepth(this->root));
		}
	}
}

Node* MCTS::select(Node* root){
	Node* current = root;
    while (!current->is_leaf()) {
		// printf("Traversing...\n");
        // get the max Q+U value
		Node* best_child = nullptr;
		float best_score = -1;
		for (int i = 0; i < (int)current->getChildren().size(); i++) {
			Node* child = current->getChildren()[i];
			float score = child->getPUCTScore();
			if (score >= best_score) {
				best_child = child;
				best_score = score;
			}
		}
		if (best_child == nullptr) {
			break;
		}
		current = best_child;
    }
	return current;
}

float MCTS::expand(Node* node){
	// printf("Expanding...\n");
	
	// convert board to input state
	Board board = Board(node->getFen());
	std::array<boolBoard, 19> inputState = board.boardToInput();
	
	// outputs
	std::array<floatBoard, 73> output_probs;
	float output_value = 0.0;

	// send input to neural network
	this->nn.predict(inputState, output_probs, output_value);

	// output to moves
	std::vector<thc::Move> legal_moves;
	board.getLegalMoves(legal_moves);
	
	std::map<thc::Move, float> moveProbs = board.outputProbsToMoves(output_probs, legal_moves);

	// std::cout << "legal moves: " << legal_moves.size() << std::endl;

	// add nodes for every move
	for (int i = 0; i < (int)legal_moves.size(); i++) {
		// make the move on the board
		std::string new_fen = board.makeMove(legal_moves[i]);

		// get the move probability for this move
		float prior = moveProbs[legal_moves[i]];
		
		// create a new node
		Node* child = new Node(new_fen, node, prior);
		// add the new node to the children of the current node
		node->add_child(child);

		// undo move
		board.undoMove(legal_moves[i]);
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

int MCTS::getTreeDepth(Node* root) {
	// recursive function of getting height of the tree
	if (root->is_leaf()) {
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