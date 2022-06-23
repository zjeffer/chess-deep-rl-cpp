#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
#include "neuralnet.hh"
#include "utils.hh"
#include "common.hh"

void signal_handling(int signal) {
	LOG(INFO) << "Signal " << signal << " received. Quitting...";
	g_running = false;
}


int playGame(int amount_of_sims, Agent* white, Agent* black){
	Environment env = Environment();
	Game game = Game(amount_of_sims, env, white, black);
	return game.playGame();	
}

void playContinuously(std::string networkPath, int amount_of_sims, int parallel_games){
	// create the neural network
	// TODO: is it necessary to load the network every time?
	NeuralNetwork* nn = new NeuralNetwork(networkPath);

	struct Winners {
		int white = 0;
		int black = 0;
		int draw = 0;
	};

	struct Winners winners;
	
	// TODO: make parallel games possible

	while (g_running){
		Agent* white = new Agent("white", nn);
		Agent* black = new Agent("black", nn);
		
		int winner = playGame(amount_of_sims, white, black);
		std::cout << "\n\n\n";
		if (winner == 1) {
			LOG(INFO) << "White won";
			winners.white++;
		} else if (winner == -1) {
			LOG(INFO) << "Black won";
			winners.black++;
		} else {
			LOG(INFO) << "Draw";
			winners.draw++;
		}
		LOG(INFO) << "\nCurrent score: \n";
		LOG(INFO) << "White: " << winners.white;
		LOG(INFO) << "Black: " << winners.black;
		LOG(INFO) << "Draw: " << winners.draw;
		LOG(INFO) << "\n\n\n";
	}
}

void playPosition(std::string fen, int amount_of_sims){
	std::cout << "Creating environment with position: " << fen << std::endl;
	Environment env = Environment(fen);
	std::cout << "Creating NN" << std::endl;
	NeuralNetwork* nn = new NeuralNetwork("models/model.pt", false);
	Agent* white = new Agent("white", nn);
	Agent* black = new Agent("black", nn);
	std::cout << "Creating game" << std::endl;
	Game game = Game(amount_of_sims, env, white, black);
	game.getEnvironment()->printBoard();
	game.playMove();
}

int main(int argc, char** argv) {
	// signal handling
	signal(SIGINT, signal_handling);
	signal(SIGTERM, signal_handling);

	auto logger = std::make_unique<Logger>();

	// set random seed
	g_generator.seed(std::random_device{}());
	LOG(DEBUG) << "Test random value: " << g_generator();

	int amount_of_sims = 50;
	int parallel_games = 1;
	if (argc >= 2) {
		try {
			amount_of_sims = std::stoi(argv[1]);
			if (argc == 3){
				parallel_games = std::stoi(argv[2]);
			}
		} catch (std::invalid_argument) {
			LOG(FATAL) << "Invalid argument";
			exit(EXIT_FAILURE);
		}
	}

	// test mcts simulations:
	// utils::test_MCTS();

	// test neural network input & outputs:
	// utils::test_NN("");

	// try training
	utils::test_Train("models/model.pt");

	// play chess
	// playContinuously("models/model.pt", amount_of_sims, parallel_games);

	// load tensor from file
	// utils::viewTensorFromFile("memory/game-1655150416-650752/move-000-output.pt");

	// test positions with quick mates
	// playPosition("7k/5ppp/8/8/8/6N1/1PPPPPPP/R3KBBN w - - 0 1", amount_of_sims);
	// playPosition("5k2/2p1p1pB/R1P1p1Kb/P3P1pP/P5p1/4p1p1/4PqP1/8 w - - 0 1", amount_of_sims);
	// playPosition("5r1k/7p/7K/7P/6Q1/8/8/8 w - - 0 1", amount_of_sims);

	// test mate in 4 position (advanced)
	// playPosition("3q1rk1/6n1/p2P1pPQ/1p5p/1P5P/5p1b/P4P2/3R1KR1 w - - 1 35", amount_of_sims);

	logger->destroy();
	logger.reset();

	return 0;
}