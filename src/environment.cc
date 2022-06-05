#include "environment.hh"
#include <string>
#include <iostream>
#include "utils.hh"


Environment::Environment(thc::ChessRules rules) {
	this->rules = rules;
	this->terminalState = thc::NOT_TERMINAL;
}

Environment::Environment(std::string fen): Environment(thc::ChessRules()) {
    this->rules.Forsyth(fen.c_str());
}

Environment::Environment() : Environment(thc::ChessRules()) {
	
}

thc::ChessRules* Environment::getRules() {
	return &(this->rules);
}

bool Environment::isGameOver() {
	// TODO: implement
	bool okay = this->rules.Evaluate(this->terminalState);

	if (!okay){
		G3LOG(FATAL) << "Error: Game received illegal position" << std::endl;
		exit(EXIT_FAILURE);
	}

	return this->terminalState != thc::TERMINAL::NOT_TERMINAL;
}

void Environment::reset() {
	if (!this->rules.Forsyth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")){
        G3LOG(FATAL) << "Error: Could not reset environment" << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Environment::printBoard() {
    G3LOG(INFO) << "\n" << this->rules.ToDebugStr() << "\n";
}

bool Environment::getCurrentPlayer() {
	return this->rules.WhiteToPlay();
}

std::string Environment::getFen() {
	return this->rules.ForsythPublish();
}

int Environment::getAmountOfPieces(){
    int pieces = 0;
    for (int i = 0; i < 64; i++){
        if (this->rules.squares[i] != ' '){
            pieces++;
        }
    }
    return pieces;
}

std::string Environment::makeMove(thc::Move move) {
    // check if move is empty
    if (!move.Valid()){
        G3LOG(FATAL) << "Move is not valid" << std::endl;
        exit(EXIT_FAILURE);
    }
    int amountOfPiecesBeforeMove = this->getAmountOfPieces();
	this->rules.PlayMove(move);
    int amountOfPiecesAfterMove = this->getAmountOfPieces();
    if (abs(amountOfPiecesBeforeMove - amountOfPiecesAfterMove) > 1){
        G3LOG(FATAL) << "Move did not result in correct amount of pieces" << std::endl;
        exit(EXIT_FAILURE);
    }
    return this->getFen();
}

std::string Environment::undoMove(thc::Move move) {
    this->rules.PopMove(move);
    return this->getFen();
}

void Environment::getLegalMoves(std::vector<thc::Move> &moves) {
    this->rules.GenLegalMoveList(moves);
}

torch::Tensor Environment::boardToInput() {
    torch::Tensor input = torch::zeros({119, 8, 8});

    // add all the pieces and the repitition counts (14 planes * 8 moves)
    thc::ChessRules currentBoard;
    std::memcpy(&currentBoard, &this->rules, sizeof(thc::ChessRules));
    for (int i = 0; i < (int)this->rules.history_idx; i++) {
        utils::addboardToPlanes(&input, i, &currentBoard);
        currentBoard.PopMove(currentBoard.history[currentBoard.history_idx-i-1]);
    }

    // current turn (plane 113)
    if (this->rules.WhiteToPlay()) {
        input[14*8] = torch::ones({8, 8});
    } else {
        input[14*8] = torch::zeros({8, 8});
    }

    // total move counter (plane 114)
    input[14*8 + 1] = torch::full({8, 8}, this->rules.full_move_count);

    // castling rights (planes 115-118)
    input[14*8 + 2] = torch::full({8, 8}, this->rules.wqueen == 1);
    input[14*8 + 3] = torch::full({8, 8}, this->rules.wking == 1);
    input[14*8 + 4] = torch::full({8, 8}, this->rules.bqueen == 1);
    input[14*8 + 5] = torch::full({8, 8}, this->rules.bking == 1);

    // no-progress counter (plane 119)
    input[14*8 + 6] = torch::full({8, 8}, this->rules.half_move_clock);
    
    // expand dims
    input = input.unsqueeze(0);
    return input;
}



