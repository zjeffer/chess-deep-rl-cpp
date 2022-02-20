#ifndef GAME_HH
#define GAME_HH

#include "agent.hh"
#include "environment.hh"
#include "chess/thc.hh"

#define MAX_MOVES 100
#define AMOUNT_OF_SIMULATIONS 400

struct MemoryElement {
	std::string state;
	std::vector<MoveProb> probs;
	int winner;
};

class Game {
  public:
	Game();
    Game(Environment env, Agent white, Agent black);

    int playGame();

    void play_move();

    void saveToMemory(MemoryElement element);

    void updateMemory(int winner);

	void memoryToFile();

	void reset();

  private:
	Environment env;
	
	Agent white;
	Agent black;

	thc::Move *previous_moves;

	std::vector<MemoryElement> memory;

};

#endif // GAME_HH