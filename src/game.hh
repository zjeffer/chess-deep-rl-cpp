#ifndef GAME_HH
#define GAME_HH

#include "agent.hh"
#include "environment.hh"
#include "chess/thc.hh"

#define MAX_MOVES 1000


class Game {
  public:
	/**
	 * @brief Construct a new Game object
	 * 
	 * @param simulations 
	 * @param env 
	 * @param white 
	 * @param black 
	 */
    Game(int simulations = 400, Environment env = Environment(), Agent white = Agent("white"), Agent black = Agent("black"));

	/**
	 * @brief Play one game. Returns the winner.
	 * 
	 * @return int: the winner: 1 for white, -1 for black, 0 for draw 
	 */
    int playGame();

	/**
	 * @brief Play one move.
	 * 
	 */
    void play_move();

	/**
	 * @brief Add the given MemoryElement to the current memory.
	 * 
	 * @param element 
	 */
    void saveToMemory(MemoryElement element);

	/**
	 * @brief Assign the given winner to all elements in the current memory.
	 * 
	 * @param winner 
	 */
    void updateMemory(int winner);

	/**
	 * @brief Convert the given MemoryElement to input and output tensors
	 * 
	 * @param memory_element 
	 * @param input_tensor 
	 * @param output_tensor 
	 */
	void memoryElementToTensors(MemoryElement *memory_element, torch::Tensor& input_tensor, torch::Tensor& output_tensor);

	/**
	 * @brief Write the current memory to a file.
	 * 
	 */
	void memoryToFile();

	/**
	 * @brief Reset the environment and clear the memory
	 * so a new game can be started.
	 * 
	 */
	void reset();

  private:
	// holds the amount of simulations to run every move
	int simulations;

	// the environment the agents will interact with
	Environment env;
	
	// the agent that will interact with the environment
	Agent white;
	Agent black;

	// the pair of previous moves played
	thc::Move *previous_moves;
	
	// for every move, holds the state, move probs and the eventual winner 
	std::vector<MemoryElement> memory;
	
	// a unique id for this game
	std::string game_id;
};

#endif // GAME_HH