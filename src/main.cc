#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
#include "neuralnet.hh"
#include "utils.hh"

bool is_running = true;
void signal_handling(int signal) {
	std::cout << "Signal " << signal << " received. Quitting..." << std::endl;
	is_running = false;
}

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
	NeuralNetwork nn = NeuralNetwork("models/model.pt", false);

	Environment board = Environment();
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
	cv::Mat mat = utils::tensorToMat(input.clone(), 119*8, 8);
	utils::saveCvMatToImg(mat, "tests/input.png", 128);

	torch::Tensor output = torch::zeros({4673});

	// predict
	nn.predict(input, output);

	std::cout << "predicted" << std::endl;

	// value is the last element of the output tensor
	torch::Tensor value = output.slice(1, 4672, 4673);
	std::cout << "value: " << value << std::endl;
	torch::Tensor policy = output.slice(1, 0, 4672).view({73, 8, 8});
	std::cout << "policy: " << policy.sizes() << std::endl;
	// reshape to 73x8x8
	cv::Mat img = utils::tensorToMat(policy.clone(), 73*8, 8);
	std::cout << "image: " << img.size() << std::endl;
	utils::saveCvMatToImg(img, "tests/output.png", 255);
}

void test_Train(){
	NeuralNetwork nn = NeuralNetwork();

	ChessDataSet chessDataSet = ChessDataSet("memory");
	
	auto train_set = chessDataSet.map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	int batch_size = 128;
	
	// data loader
	std::cout << "Creating data loader" << std::endl;
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);
	std::cout << "Data loader created" << std::endl;


	// optimizer
	int learning_rate = 0.2;
	torch::optim::Adam optimizer(nn.parameters(), learning_rate);

	
	nn.train(*data_loader, optimizer, train_set_size, batch_size);
}

int playGame(int amount_of_sims, Agent& white, Agent& black){
	Game game = Game(amount_of_sims, Environment(), white, black);
	return game.playGame();	
}

void playContinuously(int amount_of_sims, int parallel_games){
	// create the neural network
	// TODO: is it necessary to create a new network every time?
	NeuralNetwork* nn = new NeuralNetwork("models/model.pt");

	struct Winners {
		int white = 0;
		int black = 0;
		int draw = 0;
	};

	struct Winners winners;
	
	// TODO: make parallel games possible

	while (is_running){
		Agent white = Agent("white", nn);
		Agent black = Agent("black", nn);
		
		int winner = playGame(amount_of_sims, white, black);
		std::cout << "\n\n\n";
		if (winner == 1) {
			std::cout << "White won" << std::endl;
			winners.white++;
		} else if (winner == -1) {
			std::cout << "Black won" << std::endl;
			winners.black++;
		} else {
			std::cout << "Draw" << std::endl;
			winners.draw++;
		}
		std::cout << "\nCurrent score: \n";
		std::cout << "White: " << winners.white << std::endl;
		std::cout << "Black: " << winners.black << std::endl;
		std::cout << "Draw: " << winners.draw << std::endl;
		std::cout << "\n\n\n";
	}
}

int main(int argc, char** argv) {
	// signal handling
	signal(SIGINT, signal_handling);
	signal(SIGTERM, signal_handling);

	int amount_of_sims = 20;
	int parallel_games = 1;
	if (argc >= 2) {
		try {
			amount_of_sims = std::stoi(argv[1]);
			if (argc == 3){
				parallel_games = std::stoi(argv[2]);
			}
		} catch (std::invalid_argument) {
			std::cerr << "Invalid argument" << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	// test mcts simulations:
	// test_MCTS();

	// test neural network input & outputs:
	// test_NN();

	// try training
	// test_Train();

	// play chess
	playContinuously(amount_of_sims, parallel_games);


	return 0;
}