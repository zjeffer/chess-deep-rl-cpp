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
	G3LOG(INFO) << "Signal " << signal << " received. Quitting...";
	g_running = false;
}


int playGame(int amount_of_sims, Agent& white, Agent& black){
	Environment env = Environment();
	Game game = Game(amount_of_sims, env, white, black);
	return game.playGame();	
}

void playContinuously(std::string networkPath, int amount_of_sims, int parallel_games){
	// create the neural network
	// TODO: is it necessary to load the network every time?
	NeuralNetwork* nn = new NeuralNetwork(networkPath);

	// TODO: fix multiple loads of the NN every time a new game starts

	struct Winners {
		int white = 0;
		int black = 0;
		int draw = 0;
	};

	struct Winners winners;
	
	// TODO: make parallel games possible

	while (g_running){
		Agent white = Agent("white", nn);
		Agent black = Agent("black", nn);
		
		int winner = playGame(amount_of_sims, white, black);
		std::cout << "\n\n\n";
		if (winner == 1) {
			G3LOG(INFO) << "White won";
			winners.white++;
		} else if (winner == -1) {
			G3LOG(INFO) << "Black won";
			winners.black++;
		} else {
			G3LOG(INFO) << "Draw";
			winners.draw++;
		}
		G3LOG(INFO) << "\nCurrent score: \n";
		G3LOG(INFO) << "White: " << winners.white;
		G3LOG(INFO) << "Black: " << winners.black;
		G3LOG(INFO) << "Draw: " << winners.draw;
		G3LOG(INFO) << "\n\n\n";
	}
}

void playPosition(std::string fen, int amount_of_sims){
	std::cout << "Creating environment with position: " << fen << std::endl;
	Environment env = Environment(fen);
	std::cout << "Creating NN" << std::endl;
	NeuralNetwork nn = NeuralNetwork();
	std::cout << "Creating white agent" << std::endl;
	Agent white = Agent("white", &nn);
	std::cout << "Creating black agent" << std::endl;
	Agent black = Agent("black", &nn);
	std::cout << "Creating game" << std::endl;
	Game game = Game(amount_of_sims, env, white, black);
	game.getEnvironment()->printBoard();
	game.playMove();
	// std::cout << "Game over. Result: " << winner << std::endl;
}

int main(int argc, char** argv) {
	// signal handling
	signal(SIGINT, signal_handling);
	signal(SIGTERM, signal_handling);

	auto logger = std::make_unique<Logger>();

	int amount_of_sims = 20;
	int parallel_games = 1;
	if (argc >= 2) {
		try {
			amount_of_sims = std::stoi(argv[1]);
			if (argc == 3){
				parallel_games = std::stoi(argv[2]);
			}
		} catch (std::invalid_argument) {
			G3LOG(FATAL) << "Invalid argument";
			exit(EXIT_FAILURE);
		}
	}

	// test mcts simulations:
	// utils::test_MCTS();

	// test neural network input & outputs:
	// utils::test_NN("models/model.pt");

	// try training
	// utils::test_Train();

	// play chess
	// playContinuously("models/model.pt", amount_of_sims, parallel_games);

	playPosition("7k/5ppp/8/8/8/6N1/1PPPPPPP/R3KBBN w - - 0 1", amount_of_sims);

	logger->destroy();
	logger.reset();

	return 0;
}