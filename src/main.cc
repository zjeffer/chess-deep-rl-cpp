#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
#include "neuralnet.hh"
#include <iostream>

void test_MCTS(){
	Environment env = Environment();
	std::cout << env.getFen() << std::endl;

	// test mcts tree
	MCTS mcts = MCTS(new Node(), new NeuralNetwork());

	// run sims
	mcts.run_simulations(1000);

	// show actions of root
	Node* root = mcts.getRoot();
	std::vector<Node*> nodes = root->getChildren();
	thc::ChessRules* cr = env.getRules();
	printf("Possible moves in state %s: \n", env.getFen().c_str());
	for (int i = 0; i < (int)nodes.size(); i++) {
		printf("%s \t Prior: %f \t Q: %f \t U: %f\n", nodes[i]->getAction().NaturalOut(cr).c_str(), nodes[i]->getPrior(), nodes[i]->getQ(), nodes[i]->getUCB());
	}
}

void test_NN(){
	NeuralNetwork nn = NeuralNetwork();
	std::array<boolBoard, 19> input = {};
	std::array<floatBoard, 73> output_probs = {};
	float output_value = 0.0;
	// fill input
	Environment board = Environment();

	input = board.boardToInput();

	// print board
	for (int i = 0; i < 19; i++) {
		for (int j = 0; j < 8; j++){
			for(int k = 0; k < 8; k++){
				std::cout << input[i].board[j][k];
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	// predict
	nn.predict(input, output_probs, output_value);
	nn.predict(input, output_probs, output_value);
	nn.predict(input, output_probs, output_value);
}

int main(int argc, char** argv) {

	// test mcts simulations:
	// test_MCTS();

	// test neural network:
	// test_NN();

	// play one game
	Game game = Game();
	game.playGame();
	

	return 0;
}