#ifndef GAME_HH
#define GAME_HH

#include "agent.hh"
#include "environment.hh"
#include "chess/thc.hh"

#define MAX_MOVES 100


class Game {
  public:
    Game(int simulations = 400, Environment env = Environment(), Agent white = Agent("white"), Agent black = Agent("black"));

    int playGame();

    void play_move();

    void saveToMemory(MemoryElement element);

    void updateMemory(int winner);

	void memoryToFile();

	void reset();

  private:
	int simulations;
	Environment env;
	
	Agent white;
	Agent black;

	thc::Move *previous_moves;

	std::vector<MemoryElement> memory;
	
	std::string game_id;
};

#endif // GAME_HH