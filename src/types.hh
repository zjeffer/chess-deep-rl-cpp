#pragma once

#include <string>
#include <vector>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/torch.h>
#include "chess/thc.hh"

/**
 * @brief A single Move with its probability
 * 
 */
struct MoveProb {
	// the move
	thc::Move move;
	// the probability
	float prob;
};

/**
 * @brief Represents a single element of chess memory
 */
struct MemoryElement {
	// the fen string
	std::string state;
	// a dictionary of moves and their probabilities
	std::vector<MoveProb> probs;
	// the eventual winner of the game
	int winner;
};


