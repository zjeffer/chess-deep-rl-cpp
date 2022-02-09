#include "mcts.hh"
#include "board.hh"
#include <iostream>

void test_MCTS(){
	Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
	std::cout << board.getFen() << std::endl;

	// test mcts tree
	MCTS mcts = MCTS(new Node());

	// run sims
	mcts.run_simulations(2000);

	// show actions of root
	Node* root = mcts.getRoot();
	std::vector<Node*> nodes = root->getChildren();
	thc::ChessRules cr = board.getChessRules();
	printf("Possible moves in state %s: \n", board.getFen().c_str());
	for (int i = 0; i < (int)nodes.size(); i++) {
		printf("%s \t Prior: %f \t Q: %f \t U: %f\n", nodes[i]->getAction().NaturalOut(&cr).c_str(), nodes[i]->getPrior(), nodes[i]->getQ(), nodes[i]->getUCB());
	}
}

void test_NN(){
	NeuralNetwork nn = NeuralNetwork();
	std::array<boolBoard, 19> input = {};
	std::array<floatBoard, 73> output_probs = {};
	float output_value = 0.0;
	// fill input
	Board board = Board();

	// predict
	nn.predict(input, output_probs, output_value);
	
}

int main(int argc, char** argv) {

	// test mcts simulations:
	// test_MCTS();

	// test neural network:
	test_NN();
	

	return 0;
}