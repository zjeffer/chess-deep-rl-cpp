#include <iostream>
#include <opencv2/opencv.hpp>

#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
#include "neuralnet.hh"
#include "utils.hh"


void test_MCTS(){
	Environment env = Environment();
	std::cout << env.getFen() << std::endl;

	// test mcts tree
	MCTS mcts = MCTS(new Node(), new NeuralNetwork());

	// run sims
	mcts.run_simulations(400);

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
	std::array<floatBoard, 73> output_probs = {};
	float output_value = 0.0;
	// fill input
	Environment board = Environment();

	torch::Tensor input = board.boardToInput();

	// print board
	for (int i = 0; i < 119; i++) {
		for (int j = 0; j < 8; j++){
			for(int k = 0; k < 8; k++){
				std::cout << input[i][j][k];
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

void test_input(){
	Environment board = Environment();
	// create some moves
	std::vector<std::string> moveList = {
		"e2e4", 
		"e7e5",
		"g1f3",
		"b8c6",
		"f1c4",
		"f8c5",
		"e1g1"
	};

	// play the moves
	for (std::string moveString : moveList){
		thc::Move move;
		if (move.TerseIn(board.getRules(), moveString.c_str())){
			board.makeMove(move);
		} else {
			std::cerr << "Invalid move: " << moveString << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	// test board to input
	std::cout << "Converting board to input state" << std::endl;
	torch::Tensor input = board.boardToInput();

	// tensor to image
	std::cout << "Converting input to image" << std::endl;
	cv::Mat mat = utils::tensorToMat(input);
	utils::saveCvMatToImg(mat, "tests/output_test.png");
}

void test_Train(){
	NeuralNetwork nn = NeuralNetwork();

	auto train_set = ChessDataSet("memory").map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	int batch_size = 64;
	
	// data loader
	std::cout << "Creating data loader" << std::endl;
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), 8);
	std::cout << "Data loader created" << std::endl;

	// optimizer
	torch::optim::Adam optimizer(nn.parameters(), 0.01);

	std::cout << "Starting training" << std::endl;
	for (auto& batch: *data_loader){
		std::cout << "Batch" << std::endl;
		auto data = batch.data.to(torch::kCUDA);
		auto target = batch.target.to(torch::kCUDA);
	}
	std::cout << "Training finished" << std::endl;
	
	// nn.train(*data_loader, optimizer, train_set_size);
}

int playGame(int argc, char** argv){
	int amount_of_sims = 10;
	if (argc == 2) {
		try {
			amount_of_sims = std::stoi(argv[1]);
		} catch (std::invalid_argument) {
			std::cerr << "Invalid argument" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	// create the NN
	NeuralNetwork* nn = new NeuralNetwork();
	// create the agents, both using the same NN for selfplay
	Agent white = Agent("white", nn);
	Agent black = Agent("black", nn);

	// play one game
	Game game = Game(amount_of_sims, Environment(), white, black);
	return game.playGame();
}

int main(int argc, char** argv) {

	// test mcts simulations:
	// test_MCTS();

	// test neural network:
	// test_NN();

	// try training
	// test_Train();

	// test board to input state
	// test_input();

	// play chess
	int winner = playGame(argc, argv);


	return 0;
}