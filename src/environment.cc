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
		std::cerr << "Error: Game received illegal position" << std::endl;
		return false;
	}

	return this->terminalState != thc::TERMINAL::NOT_TERMINAL;
}

void Environment::reset() {
	this->rules.Forsyth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
}

void Environment::printBoard() {
    std::cout << std::endl;
	char* board = this->rules.squares;
	for (int i = 0; i < 64; i++) {
		std::cout << board[i] << ".";
		if (i % 8 == 7) {
			std::cout << std::endl;
		}
	}
    std::cout << std::endl;
}

bool Environment::getCurrentPlayer() {
	return this->rules.WhiteToPlay();
}

std::string Environment::getFen() {
	return this->rules.ForsythPublish();
}

std::string Environment::makeMove(thc::Move move) {
	this->rules.PlayMove(move);
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
    torch::Tensor input = torch::zeros({119, 8, 8}, torch::kUInt8);

    std::cout << "History size: " << (int)this->rules.history_idx << std::endl;

    thc::ChessRules currentBoard;
    std::memcpy(&currentBoard, &this->rules, sizeof(thc::ChessRules));

    addboardToPlanes(&input, 0, &currentBoard);
    for (int i = 1; i <= (int)this->rules.history_idx; i++) {
        int board_index = currentBoard.history_idx - i;
        addboardToPlanes(&input, i, &currentBoard);
        currentBoard.PopMove(currentBoard.history[board_index]);
    }

    std::cout << currentBoard.ToDebugStr() << std::endl;

    // current turn
    if (this->rules.WhiteToPlay()) {
        input[14*8] = torch::ones({8, 8});
    } else {
        input[14*8] = torch::zeros({8, 8});
    }

    // total move count
    input[14*8 + 1] = torch::full({8, 8}, this->rules.full_move_count);

    // castling
    input[14*8 + 2] = torch::full({8, 8}, this->rules.wqueen == 1);
    input[14*8 + 3] = torch::full({8, 8}, this->rules.wking == 1);
    input[14*8 + 4] = torch::full({8, 8}, this->rules.bqueen == 1);
    input[14*8 + 5] = torch::full({8, 8}, this->rules.bking == 1);

    input[14*8 + 6] = torch::full({8, 8}, this->rules.half_move_clock);
    
    return input;
}

std::map<thc::Move, float> Environment::outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves) {
    std::map<thc::Move, float> moves = {};

    for (int i = 0; i < (int)legalMoves.size(); i++) {
        std::tuple<int, int, int> tpl = this->moveToPlaneIndex(legalMoves[i]);
        moves[legalMoves[i]] = outputProbs[std::get<0>(tpl)].board[std::get<1>(tpl)][std::get<2>(tpl)];
    }
    return moves;
}


std::array<floatBoard, 73> Environment::movesToOutputProbs(std::vector<MoveProb> moves){
    std::array<floatBoard, 73> output;
    for (MoveProb move : moves){
        std::tuple<int, int, int> tpl = this->moveToPlaneIndex(move.move);
        output[std::get<0>(tpl)].board[std::get<1>(tpl)][std::get<2>(tpl)] = move.prob;
    }
    return output;
}

std::tuple<int, int, int> Environment::moveToPlaneIndex(thc::Move move){
    char piece = this->getRules()->squares[move.src];
    int plane_index = -1;
    int direction = 0;

    if (piece == ' ') {
        printf("No piece on that square!\n");
        exit(EXIT_FAILURE);
    }

    if (move.special >= thc::SPECIAL_PROMOTION_ROOK and
        move.special <= thc::SPECIAL_PROMOTION_KNIGHT) {
        // get directions
        direction = Mapper::getUnderpromotionDirection(move.src, move.dst);

        // get type of special move
        int promotion_type;
        if (move.special == thc::SPECIAL_PROMOTION_KNIGHT) {
            promotion_type = UnderPromotion::KNIGHT;
        } else if (move.special == thc::SPECIAL_PROMOTION_BISHOP) {
            promotion_type = UnderPromotion::BISHOP;
        } else if (move.special == thc::SPECIAL_PROMOTION_ROOK) {
            promotion_type = UnderPromotion::ROOK;
        } else {
            printf("Unhandled promotion type: %d\n", move.special);
        }

        plane_index = mapper[promotion_type][1 - direction];
    } else if (tolower(this->getRules()->squares[move.src]) == 'n') {
        // get the correct knight move
        direction = Mapper::getKnightDirection(move.src, move.dst);
        plane_index = mapper[KnightMove::NORTH_LEFT + direction][0];
    } else {
        // get the correct direction
        std::tuple<int, int> tuple =
            Mapper::getQueenDirection(move.src, move.dst);
        plane_index = mapper[std::get<0>(tuple)][std::get<1>(tuple)];
    }

    if (plane_index < 0 or plane_index > 72) {
        printf("Plane index: %d\n", plane_index);
        perror("Plane index out of bounds!");
        exit(EXIT_FAILURE);
    }

    int row = move.src / 8;
    int col = move.src % 8;
    return std::make_tuple(plane_index, row, col);
}