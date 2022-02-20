#ifndef ENVIRONMENT_HH
#define ENVIRONMENT_HH

#include "chess/thc.hh"
#include <array>
#include "map"
#include "mapper.hh"

struct boolBoard {
	std::array<std::array<bool, 8>, 8> board {};
};

struct floatBoard {
	std::array<std::array<float, 8>, 8> board {};
};

class Environment {
	public:
		Environment(thc::ChessRules rules);
		Environment(std::string fen);
		Environment();

		void reset();

		bool isGameOver();

		void printBoard();

		thc::TERMINAL terminalState;

		bool getCurrentPlayer();

		std::string getFen();

		std::string makeMove(thc::Move move);
		std::string undoMove(thc::Move move);

		thc::ChessRules* getRules();

		std::array<boolBoard, 19> boardToInput();

		std::map<thc::Move, float> outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves);

		void getLegalMoves(std::vector<thc::Move> &moves);

		

	private:
		thc::ChessRules rules;
		

};

#endif // ENVIRONMENT_HH