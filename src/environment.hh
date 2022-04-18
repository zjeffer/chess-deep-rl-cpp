#ifndef ENVIRONMENT_HH
#define ENVIRONMENT_HH

#include <array>
#include <map>
#include "chess/thc.hh"
#include "types.hh"


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

		/**
		 * @brief Convert the current board to an input interpretable by the model
		 * 
		 * @return std::array<boolBoard, 19> 
		 */
		torch::Tensor boardToInput();

		/**
		 * @brief Convert a list of moves to a policy output vector
		 * 
		 * @param moves 
		 * @return std::array<floatBoard, 73> 
		 */
		std::array<floatBoard, 73> movesToOutputProbs(std::vector<MoveProb> moves);

		/**
		 * @brief Convert a move to a plane index, the row, and column on the board
		 * 
		 * @return std::tuple<int, int, int> 
		 */
		std::tuple<int, int, int> moveToPlaneIndex(thc::Move);

		void getLegalMoves(std::vector<thc::Move> &moves);

		

	private:
		thc::ChessRules rules;

};

#endif // ENVIRONMENT_HH