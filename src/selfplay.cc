#include "selfplay.hh"

int SelfPlay::testThread() {
    LOG(INFO) << "Test thread";
    return 0;
}

int SelfPlay::playGame(int amount_of_sims, Agent* white, Agent* black){
	Environment env = Environment();
	Game game = Game(amount_of_sims, env, white, black);
	return game.playGame();	
}

void SelfPlay::playContinuously(std::string networkPath, int amount_of_sims, int parallel_games) {
    // create the neural network
    // TODO: is it necessary to load the network every time?
    NeuralNetwork *nn = new NeuralNetwork(networkPath);

    struct Winners {
        int white = 0;
        int black = 0;
        int draw = 0;
    };

    struct Winners winners;

    // TODO: make parallel games possible

    while (g_running) {
        Agent *white = new Agent("white", nn);
        Agent *black = new Agent("black", nn);

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

void SelfPlay::playPosition(std::string fen, int amount_of_sims){
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