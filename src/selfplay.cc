#include "selfplay.hh"

int SelfPlay::playGame(int amount_of_sims, Agent* white, Agent* black){
	Environment env = Environment();
	Game game = Game(amount_of_sims, env, white, black);
	return game.playGame();	
}

void SelfPlay::playContinuously(NeuralNetwork* nn, int amount_of_sims, int parallel_games) {
    struct Winners {
        int white = 0;
        int black = 0;
        int draw = 0;
    };

    struct Winners winners;

    while (g_running && g_isSelfPlaying) {
        Agent *white = new Agent("white", nn);
        Agent *black = new Agent("black", nn);

        int winner = playGame(amount_of_sims, white, black);
        std::cout << "\n\n\n";

        if (!g_isSelfPlaying) {
            break;
        }

        if (winner == 1) {
            // g_mainWindow->print("White won!");
            winners.white++;
        } else if (winner == -1) {
            // g_mainWindow->print("Black won!");
            winners.black++;
        } else {
            // g_mainWindow->print("Draw!");
            winners.draw++;
        }
        // g_mainWindow->print("\nCurrent score: \n");
        // g_mainWindow->print("White: " + winners.white);
        // g_mainWindow->print("Black: " + winners.black);
        // g_mainWindow->print("Draw: " + winners.draw);
        // g_mainWindow->print("\n\n\n");
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