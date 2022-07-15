#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include <iostream>
#include <signal.h>
#include <opencv2/opencv.hpp>

#include "ui/mainwindow.hh"
#include <QApplication>
#include <thread>

#include "common.hh"

MainWindow* g_mainWindow = nullptr;

void signal_handling(int signal) {
	LOG(INFO) << "Signal " << signal << " received. Quitting...";
	g_running = false;

	if (g_mainWindow != nullptr) {
		g_mainWindow->close();
	}
}


int main(int argc, char** argv) {
	// signal handling
	signal(SIGINT, signal_handling);
	signal(SIGTERM, signal_handling);

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

	QApplication a(argc, argv);
    g_mainWindow = new MainWindow();
    g_mainWindow->show();

	// test mcts simulations:
	// utils::test_MCTS();

	// test neural network input & outputs:
	// utils::test_NN("models/model_2022-07-03_13-43-24_trained.pt");

	// try training
	// utils::test_Train("models/model.pt");

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

    int return_code = a.exec();

	logger->destroy();
	logger.reset();

	return return_code;
}