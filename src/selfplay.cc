#include "selfplay.hh"

int SelfPlay::playGame(int amount_of_sims, Agent* white, Agent* black){
	Environment env = Environment();
	Game game = Game(amount_of_sims, &env, white, black);
	return game.playGame();	
}

void SelfPlay::playContinuously(const std::shared_ptr<NeuralNetwork> &nn, int amount_of_sims) {
    struct Winners {
        int white = 0;
        int black = 0;
        int draw = 0;
    };

    struct Winners winners;

    while (g_Running && g_IsSelfPlaying) {
        Agent *white = new Agent("white", nn);
        Agent *black = new Agent("black", nn);

        int winner = playGame(amount_of_sims, white, black);
        std::cout << "\n\n\n";

        if (!g_IsSelfPlaying) {
            break;
        }

        if (winner == 1) {
            LOG(INFO) << "White won!";
            winners.white++;
        } else if (winner == -1) {
            LOG(INFO) << "Black won!";
            winners.black++;
        } else {
            LOG(INFO) << "Draw!";
            winners.draw++;
        }
        LOG(INFO) << "\nCurrent score: \n";
        LOG(INFO) << "\tWhite: " << winners.white;
        LOG(INFO) << "\tBlack: " << winners.black;
        LOG(INFO) << "\tDraw: " << winners.draw;
        LOG(INFO) << "\n\n\n";
    }
}

void SelfPlay::playPosition(const std::shared_ptr<NeuralNetwork> &nn, std::string fen, int amount_of_sims){
	std::cout << "Creating environment with position: " << fen << std::endl;
	Environment env = Environment(fen);
	Agent* white = new Agent("white", nn);
	Agent* black = new Agent("black", nn);
	std::cout << "Creating game" << std::endl;
	Game game = Game(amount_of_sims, &env, white, black);
	game.getEnvironment()->printBoard();
	game.playMove();
    delete white;
    delete black;
}