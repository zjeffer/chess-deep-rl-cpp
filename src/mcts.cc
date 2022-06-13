#include <stdio.h>
#include <chrono>
#include <iostream>

#include "common.hh"
#include "chess/thc.hh"
#include "mcts.hh"
#include "tqdm.h"
#include "utils.hh"


MCTS::MCTS(Node* root, NeuralNetwork* nn) {
	this->root = root;
	this->nn = nn;
}

MCTS::~MCTS() {
	delete this->root;
}


void MCTS::run_simulations(int num_simulations) {
	G3LOG(INFO) << "Running " << num_simulations << " simulations...";

	// add dirichlet noise to the root node
	float value = expand(this->root);
	// utils::addDirichletNoise(this->root);
	for(Node *node : this->root->getChildren()) {
		node->setPrior(0.1);
	}

	tqdm bar;
	for (int i = 0; i < num_simulations && g_running; i++) {
		bar.progress(i, num_simulations);
		// selection
		Node* leaf = select(this->root);

		// expansion and evaluation
		value = expand(leaf);

		// backpropagation
		backpropagate(leaf, value);
	}
	std::cout << std::endl;
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
		G3LOG(INFO) << current->getFen();
		G3LOG(INFO) << "# Children: " << children.size();
		for (int i = 0; i < (int)children.size(); i++) {
			G3LOG(DEBUG) << "Child " << children[i]->getAction().TerseOut() << ": " << children[i]->getQ() << " + " << children[i]->getUCB() << ". Prior: " << children[i]->getPrior();
			Node* child = children[i];
			float score = child->getPUCTScore();
			if (score > best_score) {
				best_child = child;
				best_score = score;
			}
		}
		if (best_child == nullptr) {
			G3LOG(FATAL) << "Error: best_child is null";
			exit(EXIT_FAILURE);
		}
		current = best_child;
		G3LOG(WARNING) << "Selected child " << current->getAction().TerseOut() << " with score: " << best_score;
    }
	auto stop = std::chrono::high_resolution_clock::now();
	// G3LOG(DEBUG << "Selection: " << std::chrono::duration_cast<std::chrono::microseconds>(stop - start_time).count() << " microseconds for " << traversals << " traversals";

	return current;
}

float MCTS::expand(Node* node){
	// convert board to input state
	Environment env = Environment(node->getFen());
	torch::Tensor inputState = env.boardToInput();
	
	// output of neural network
	torch::Tensor output;
	
	// send input to neural network
	this->nn->predict(inputState, output);
	// get policy and value
	torch::Tensor output_policy = output.slice(1, 0, 4672).view({73, 8, 8});
	float output_value = output.slice(1, 4672, 4673).item<float>();

	// output to moves
	std::vector<thc::Move> legal_moves;
	env.getLegalMoves(legal_moves);

	if (legal_moves.size() == 0) {
		// game is finished in this node, calculate value
		G3LOG(INFO) << "No legal moves in this node: " << node->getFen();
		if (env.isGameOver()) {
			switch(env.terminalState) {
				case thc::TERMINAL_BCHECKMATE:
					G3LOG(DEBUG) << "Checkmate, white wins! Node: " << node->getFen();
					return 1.0;
				case thc::TERMINAL_WCHECKMATE:
					G3LOG(DEBUG) << "Checkmate, black wins! Node: " << node->getFen();
					return -1.0;
				case thc::TERMINAL_BSTALEMATE:
				case thc::TERMINAL_WSTALEMATE:
					G3LOG(DEBUG) << "Game is over by draw in this node: " << node->getFen();
					return 0.0;
				default:
					G3LOG(WARNING) << "Unknown terminal state: " << env.terminalState;
					exit(EXIT_FAILURE);
			}
		} else {
			G3LOG(WARNING) << "Game is not over but no legal moves in this node: " << node->getFen();
			exit(EXIT_FAILURE);
		}
	}
	
	std::map<thc::Move, float> moveProbs = utils::outputProbsToMoves(output_policy, legal_moves);

	// add nodes for every move
	for (int i = 0; i < (int)legal_moves.size(); i++) {
		thc::Move move = legal_moves[i];
		// make the move on the board
		std::string new_fen = env.makeMove(move);

		// get the move probability for this move
		float prior = moveProbs[move];
		
		// create a new node
		Node* child = new Node(new_fen, node, move, prior);
		// add the new node to the children of the current node
		node->add_child(child);

		// undo move
		env.undoMove(move);
	}
	return output_value;
}


void MCTS::backpropagate(Node* node, float value){
	while (node != nullptr) {
		// printf("Backpropagating...\n");
		node->incrementVisit();
		if (node->getPlayer()) {
			node->setValue(value);
		} else {
			node->setValue(1-value);
		}
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