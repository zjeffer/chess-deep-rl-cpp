#pragma once

#include <random>
#include "agent.hh"
#include "environment.hh"
#include "chess/thc.hh"
#include "node.hh"
#include "utils.hh"
#include "common.hh"

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
    Game(int simulations, Environment* env, Agent* white, Agent* black);

	~Game();

	/**
	 * @brief Play one game. Returns the winner.
	 * 
	 * @return int: the winner: 1 for white, -1 for black, 0 for draw 
	 */
    int playGame(bool stochastic = true);

	/**
	 * @brief Play one move.
	 * 
	 */
    void playMove();

	/**
	 * @brief Using a uniform random distribution, select the best move
	 * 
	 * @param probs 
	 * @return thc::Move 
	 */
	thc::Move getBestMoveStochastic(std::vector<MoveProb> &probs);

	/**
	 * @brief Get the best move based on the visit count of all moves
	 * 
	 * @param probs 
	 * @return thc::Move 
	 */
	thc::Move getBestMoveDeterministic(std::vector<MoveProb> &probs);

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

	Environment* getEnvironment();

  private:
	// holds the amount of simulations to run every move
	int m_Simulations;

	bool m_Stochastic = true;

	// the environment the agents will interact with
	Environment* m_Env;
	
	// the agent that will interact with the environment
	Agent* m_White;
	Agent* m_Black;

	// the pair of previous moves played
	thc::Move *m_Previous_moves;
	
	// for every move, holds the state, move probs and the eventual winner 
	std::vector<MemoryElement> m_Memory;
	
	// a unique id for this game
	std::string m_Game_id;

	// for random generation
	std::uniform_int_distribution<int> m_Dist;
};

