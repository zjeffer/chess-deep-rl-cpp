#include "environment.hh"

#include <iostream>

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
	return &this->rules;
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
		std::cout << board[i] << " ";
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
	this->rules.PushMove(move);
    return this->getFen();
}

std::string Environment::undoMove(thc::Move move) {
    this->rules.PopMove(move);
    return this->getFen();
}

void Environment::getLegalMoves(std::vector<thc::Move> &moves) {
    this->rules.GenLegalMoveList(moves);
}


std::array<boolBoard, 19> Environment::boardToInput() {
    std::array<boolBoard, 19> input{};

    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            // player's turn
            input[0].board[i][j] = this->getRules()->WhiteToPlay();
            // castling rights
            input[1].board[i][j] = this->getRules()->wqueen_allowed();
            input[2].board[i][j] = this->getRules()->wking_allowed();
            input[3].board[i][j] = this->getRules()->bqueen_allowed();
            input[4].board[i][j] = this->getRules()->bking_allowed();
            // repitition counter
            input[5].board[i][j] = this->getRules()->GetRepetitionCount() > 3;
        }
    }

    // get pieces for white
    std::string fen = getFen();
    int i = 0;
    int row = 0;
    int col = 0;
    while (fen[i] != ' ') {
        if (fen[i] == '/') {
            row++;
            col = 0;
            i++;
            continue;
        }

        if (fen[i] == 'P' || fen[i] == 'p') {
            if (fen[i] == 'P') {
                input[6].board[row][col] = true;
            } else {
                input[12].board[row][col] = true;
            }
        } else if (fen[i] == 'N' || fen[i] == 'n') {
            if (fen[i] == 'N') {
                input[7].board[row][col] = true;
            } else {
                input[13].board[row][col] = true;
            }
        } else if (fen[i] == 'B' || fen[i] == 'b') {
            if (fen[i] == 'B') {
                input[8].board[row][col] = true;
            } else {
                input[14].board[row][col] = true;
            }
        } else if (fen[i] == 'R' || fen[i] == 'r') {
            if (fen[i] == 'R') {
                input[9].board[row][col] = true;
            } else {
                input[15].board[row][col] = true;
            }
        } else if (fen[i] == 'Q' || fen[i] == 'q') {
            if (fen[i] == 'Q') {
                input[10].board[row][col] = true;
            } else {
                input[16].board[row][col] = true;
            }
        } else if (fen[i] == 'K' || fen[i] == 'k') {
            if (fen[i] == 'K') {
                input[11].board[row][col] = true;
            } else {
                input[17].board[row][col] = true;
            }
        } else {
            // number
            input[18].board[row][col] = false;
        }
        i++;
        col++;
    }

    return input;
}

std::map<thc::Move, float> Environment::outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves) {
    std::map<thc::Move, float> moves = {};

    for (int i = 0; i < (int)legalMoves.size(); i++) {
        thc::Move move = legalMoves[i];

        char piece = this->getRules()->squares[move.src];
        int plane_index = 0;
        int direction = 0;

        if (piece == ' ') {
            perror("No piece on that square!");
            exit(1);
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
            exit(1);
        }

        int row = move.src / 8;
        int col = move.src % 8;
        // create moveProb obj
        moves[move] = outputProbs[plane_index].board[row][col];
    }

    return moves;
}