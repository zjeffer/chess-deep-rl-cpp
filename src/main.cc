#include "mcts.hh"
#include "board.hh"
#include <iostream>

int main(int argc, char** argv) {

	Board board = Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
	std::cout << board.getFen() << std::endl;
	// std::array<boolBoard, 19> inputState = board.boardToInput();

	// // print inputState array
	// for (int i = 0; i < 19; i++){
	// 	for (int j = 0; j < 8; j++){
	// 		for (int k = 0; k < 8; k++){
	// 			std::cout << inputState[i].board[j][k];
	// 		}
	// 		std::cout << std::endl;
	// 	}
	// 	std::cout << std::endl;
	// }

	// test mcts tree
	MCTS mcts = MCTS(new Node());

	// run sims
	mcts.run_simulations(10000);

	return 0;
}