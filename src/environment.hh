#pragma once

#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

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

		void printDrawType(thc::DRAWTYPE drawType);

		void printBoard();

		thc::TERMINAL terminalState;

		bool getCurrentPlayer() const;

		const std::string& getFen();

		int getAmountOfPieces();

		std::string makeMove(thc::Move move);
		std::string pushMove(thc::Move move);
		std::string undoMove(thc::Move move);

		thc::ChessRules* getRules();

		/**
		 * @brief Convert the current board to an input interpretable by the model
		 * 
		 * @return torch::Tensor 
		 */
		torch::Tensor boardToInput();

		/**
		 * @brief Convert a list of moves to a policy output vector
		 * 
		 * @param moves 
		 * @return torch::Tensor
		 */
		torch::Tensor movesToOutputProbs(std::vector<MoveProb> moves);

		/**
		 * @brief Convert a move to a plane index, the row, and column on the board
		 * 
		 * @return std::tuple<int, int, int> 
		 */
		std::tuple<int, int, int> moveToPlaneIndex(thc::Move);

		void getLegalMoves(std::vector<thc::Move> &moves);

		

	private:
		thc::ChessRules m_ChessRules;
		std::string m_Fen = "";

};

