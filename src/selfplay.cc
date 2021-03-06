#include "selfplay.hh"
#include "ui/mainwindow.hh"

int SelfPlay::playGame(int amount_of_sims, Agent* white, Agent* black, MainWindow* mainWindow){
	Environment env = Environment();
	Game game = Game(amount_of_sims, &env, white, black, mainWindow);
	return game.playGame();	
}

void SelfPlay::playContinuously(const std::shared_ptr<NeuralNetwork> &nn, int amount_of_sims, MainWindow* mainWindow) {
    struct Winners {
        int white = 0;
        int black = 0;
        int draw = 0;
    };

    struct Winners winners;

    while (g_running && g_isSelfPlaying) {
        Agent *white = new Agent("white", nn);
        Agent *black = new Agent("black", nn);

        int winner = playGame(amount_of_sims, white, black, mainWindow);
        std::cout << "\n\n\n";

        if (!g_isSelfPlaying) {
            break;
        }

        if (winner == 1) {
            mainWindow->print("White won!");
            winners.white++;
        } else if (winner == -1) {
            mainWindow->print("Black won!");
            winners.black++;
        } else {
            mainWindow->print("Draw!");
            winners.draw++;
        }
        mainWindow->print("\nCurrent score: \n");
        mainWindow->print("\tWhite: " + winners.white);
        mainWindow->print("\tBlack: " + winners.black);
        mainWindow->print("\tDraw: " + winners.draw);
        mainWindow->print("\n\n\n");
    }
}

void SelfPlay::playPosition(const std::shared_ptr<NeuralNetwork> &nn, std::string fen, int amount_of_sims, MainWindow* mainWindow){
	std::cout << "Creating environment with position: " << fen << std::endl;
	Environment env = Environment(fen);
	Agent* white = new Agent("white", nn);
	Agent* black = new Agent("black", nn);
	std::cout << "Creating game" << std::endl;
	Game game = Game(amount_of_sims, &env, white, black, mainWindow);
	game.getEnvironment()->printBoard();
	game.playMove();
    delete white;
    delete black;
}