#include "mcts.hh"


MCTS::MCTS(Node* root, const std::shared_ptr<NeuralNetwork> &nn) {
	m_Root = root;
	m_NN = nn;
}

MCTS::~MCTS() {
	delete m_Root;
}


void MCTS::run_simulations(int num_simulations) {
	LOG(INFO) << "Running " << num_simulations << " simulations...";

	// add dirichlet noise to the root node
	float value = expand(m_Root);
	utils::addDirichletNoise(m_Root);

	tqdm bar;
	for (int i = 0; i < num_simulations && g_Running && g_IsSelfPlaying; i++) {
		bar.progress(i, num_simulations);
		// selection
		Node* leaf = select(m_Root);

		// expansion and evaluation
		value = expand(leaf);

		// backpropagation
		backpropagate(leaf, value);
	}
	std::cout << std::endl;
	// LOG(DEBUG) << "Tree depth: " << getTreeDepth(m_Root);
}

Node* MCTS::select(Node* root){
	Node* current = root;
    while (!current->isLeaf()) {
        // get the max Q+U value
		std::vector<Node*> children = current->getChildren();
		// start with random child as best child
		if (children.size() == 0) {
			LOG(WARNING) << "No children found for node " << current->getFen();
			exit(EXIT_FAILURE);
		}
		Node* best_child = children[rand() % children.size()];
		float best_score = -1;
		for (int i = 0; i < (int)children.size(); i++) {
			Node* child = children[i];
			float score = child->getPUCTScore();
			if (score > best_score) {
				best_child = child;
				best_score = score;
			}
		}
		if (best_child == nullptr) {
			LOG(FATAL) << "Error: best_child is null";
			exit(EXIT_FAILURE);
		}
		current = best_child;
    }
	return current;
}

float MCTS::expand(Node* node){
	// convert board to input state
	Environment env = Environment(node->getFen());
	torch::Tensor inputState = env.boardToInput();


	// send input to neural network
	std::tuple<torch::Tensor, torch::Tensor> outputs = m_NN->predict(inputState);
	// get policy and value
	torch::Tensor output_policy = std::get<0>(outputs).view({73, 8, 8});
	float output_value = std::get<1>(outputs).item<float>();

	// output to moves
	std::vector<thc::Move> legal_moves;
	env.getLegalMoves(legal_moves);

	if (legal_moves.size() == 0) {
		// game is finished in this node, calculate value
		if (env.isGameOver()) {
			switch(env.terminalState) {
				case thc::TERMINAL_BCHECKMATE:
				case thc::TERMINAL_WCHECKMATE:
					return 1.0;
				case thc::TERMINAL_BSTALEMATE:
				case thc::TERMINAL_WSTALEMATE:
					return 0.0;
				default:
					LOG(WARNING) << "Unknown terminal state: " << env.terminalState;
					exit(EXIT_FAILURE);
			}
		} else {
			LOG(WARNING) << "Game is not over but no legal moves in this node: " << node->getFen();
			exit(EXIT_FAILURE);
		}
	}
	
	std::map<thc::Move, float> moveProbs = utils::outputProbsToMoves(output_policy, legal_moves);
	float sum_priors = 0.0f;
	for (auto moveProb : moveProbs) {
		sum_priors += moveProb.second;
	}

	// add nodes for every move
	for (int i = 0; i < (int)legal_moves.size(); i++) {
		thc::Move move = legal_moves[i];
		// make the move on the board
		// TODO: fix wrong half_move_clock & full_move_count
		std::string new_fen = env.pushMove(move);

		// get the move probability for this move
		float prior = moveProbs[move];
		if (sum_priors > 0.0f) {
			prior /= sum_priors;
		}
		
		// create a new node
		Node* child = new Node(new_fen, node, move, prior);
		// add the new node to the children of the current node
		node->add_child(child);

		// undo move
		env.undoMove(move);
	}
	return output_value;
}


void MCTS::backpropagate(Node* node, float result){
	bool player = node->getPlayer();
	while (node != nullptr) {
		node->incrementVisit();
		float value = node->getValue();
		if (node->getPlayer() == player) {
			value += result;
		} else {
			value -= result;
		}
		node->setValue(value);
		node = node->getParent();
	}
}


Node* MCTS::getRoot() {
	return m_Root;
}

void MCTS::setRoot(Node* newRoot){
	m_Root = newRoot;
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