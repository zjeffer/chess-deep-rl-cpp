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
	Game game = Game(amount_of_sims, Environment(), white, black);
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

void playPosition(std::string fen){
	Environment env = Environment(fen);
	Game game = Game(400, env);
	game.getEnvironment()->printBoard();
	// list moves
	std::vector<thc::Move> moves;
	game.getEnvironment()->getLegalMoves(moves);
	for (thc::Move move : moves){
		G3LOG(INFO) << move.NaturalOut(game.getEnvironment()->getRules())
		<< " => " << move.src << "-" << move.dst << " => " << move.TerseOut();
	}

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
	playContinuously("models/model.pt", amount_of_sims, parallel_games);

	// playPosition("r2q1b2/pbPkp1pr/2n2p1n/1p3P1p/P2P2P1/8/1PPQ3P/RNBK1BNR b - - 0 13");

	// utils::testBug();

	logger->destroy();
	logger.reset();

	return 0;
}