#pragma once

#include <string>
#include <vector>
#include "chess/thc.hh"

/**
 * @brief A single Move with its probability
 * 
 */
struct MoveProb {
	~MoveProb() {};

	// the move
	thc::Move move;
	// the probability
	float prob;
};

/**
 * @brief Represents a single element of chess memory
 */
struct MemoryElement {
	~MemoryElement() {};

	// the fen string
	std::string state;
	// a dictionary of moves and their probabilities
	std::vector<MoveProb> probs;
	// the eventual winner of the game
	int winner;
};

struct LossHistory {
	~LossHistory() {};

	int historySize = 0;
	int batch_size = 0;
	int data_size = 0;
	std::vector<float> losses = std::vector<float>();
	std::vector<float> values  = std::vector<float>();
	std::vector<float> policies = std::vector<float>();
};
