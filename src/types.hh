#ifndef TYPES_HH
#define TYPES_HH

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

/**
 * @brief 8x8 board of booleans
 */
struct boolBoard {
	std::array<std::array<bool, 8>, 8> board {};
};

/**
 * @brief 8x8 board of floats
 */
struct floatBoard {
	std::array<std::array<float, 8>, 8> board {};
};

#endif // TYPES_HH