#ifndef board_hh
#define board_hh

#include "chess/thc.hh"
#include "mapper.hh"
#include <string>
#include <array>

struct boolBoard {
	std::array<std::array<bool, 8>, 8> board {};
};

struct floatBoard {
	std::array<std::array<float, 8>, 8> board {};
};

class Board {
	public:
		Board(std::string fen);
		Board();
		~Board();

		std::string makeMove(thc::Move move);
		std::string undoMove(thc::Move move);

		std::string getFen();

		std::array<boolBoard, 19> boardToInput();

		void getLegalMoves(std::vector<thc::Move> &moves);

		std::map<thc::Move, float> outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves);

		thc::ChessRules getChessRules();

	private:

		thc::ChessRules chessRules;

		

};

#endif /* board_hh */