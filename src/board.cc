#include "board.hh"
#include <chrono>
#include <iostream>

Board::Board(std::string fen) {
    this->chessRules = thc::ChessRules();
    this->chessRules.Forsyth(fen.c_str());
}

Board::~Board() {}

std::string Board::getFen() {
    return this->chessRules.ForsythPublish();
}

std::string Board::makeMove(thc::Move move) {
    // TODO: implement
    this->chessRules.PlayMove(move);
    return this->getFen();
}

std::string Board::undoMove(thc::Move move) {
    // TODO: implement
    this->chessRules.PopMove(move);
    return this->getFen();
}

void Board::getLegalMoves(std::vector<thc::Move> &moves) {
    std::vector<bool> check;
    std::vector<bool> mate;
    std::vector<bool> stalemate;

    this->chessRules.GenLegalMoveList(moves, check, mate, stalemate);
    // TODO: also use the other outputs
}


std::array<boolBoard, 19> Board::boardToInput(){
    auto start_time = std::chrono::high_resolution_clock::now();
    std::array<boolBoard, 19> input {};

    
    for (int i = 0; i < 8; i++){
        for (int j = 0; j < 8; j++){
            // player's turn
            input[0].board[i][j] = this->chessRules.WhiteToPlay();
            // castling rights
            input[1].board[i][j] = this->chessRules.wqueen_allowed();
            input[2].board[i][j] = this->chessRules.wking_allowed();
            input[3].board[i][j] = this->chessRules.bqueen_allowed();
            input[4].board[i][j] = this->chessRules.bking_allowed();
            // repitition counter
            input[5].board[i][j] = this->chessRules.GetRepetitionCount() > 3;
        }
    }

    // get pieces for white
    std::string fen = getFen();\
    int i = 0;
    int row = 0;
    int col = 0;
    while (fen[i] != ' '){ 
        if (fen[i] == '/'){
            row++;
            col = 0;
            i++;
            continue;
        }

        if (fen[i] == 'P' || fen[i] == 'p'){
            if (fen[i] == 'P'){
                input[6].board[row][col] = true;
            }
            else{
                input[12].board[row][col] = true;
            }
        } else if (fen[i] == 'N' || fen[i] == 'n'){
            if (fen[i] == 'N'){
                input[7].board[row][col] = true;
            }
            else{
                input[13].board[row][col] = true;
            }
        } else if (fen[i] == 'B' || fen[i] == 'b'){
            if (fen[i] == 'B'){
                input[8].board[row][col] = true;
            }
            else{
                input[14].board[row][col] = true;
            }
        } else if (fen[i] == 'R' || fen[i] == 'r'){
            if (fen[i] == 'R'){
                input[9].board[row][col] = true;
            }
            else{
                input[15].board[row][col] = true;
            }
        } else if (fen[i] == 'Q' || fen[i] == 'q'){
            if (fen[i] == 'Q'){
                input[10].board[row][col] = true;
            }
            else{
                input[16].board[row][col] = true;
            }
        } else if (fen[i] == 'K' || fen[i] == 'k'){
            if (fen[i] == 'K'){
                input[11].board[row][col] = true;
            }
            else{
                input[17].board[row][col] = true;
            }
        } else {
            // number
            input[18].board[row][col] = false;
        }
        i++;
        col++;
    }

    auto stop_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_time - start_time);
    // std::cout << "Time to convert board to input: " << duration.count() << "us" << std::endl;

    return input;
}

std::map<thc::Move, float> Board::outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves){
    std::map<thc::Move, float> moves = {};

    for (int i = 0; i < (int)legalMoves.size(); i++){
        thc::Move move = legalMoves[i];

        char piece = this->chessRules.squares[move.src];
        int plane_index = 0;
        int direction = 0;

        if (piece == ' '){
            perror("No piece on that square!");
            exit(1);
        }

        if (move.special >= thc::SPECIAL_PROMOTION_ROOK and move.special <= thc::SPECIAL_PROMOTION_KNIGHT){
            // get directions
            direction = Mapper::getUnderpromotionDirection(move.src, move.dst);

            // get type of special move
            int promotion_type;
            if (move.special == thc::SPECIAL_PROMOTION_KNIGHT){
                promotion_type = UnderPromotion::KNIGHT;
            } else if (move.special == thc::SPECIAL_PROMOTION_BISHOP){
                promotion_type = UnderPromotion::BISHOP;
            } else if (move.special == thc::SPECIAL_PROMOTION_ROOK){
                promotion_type = UnderPromotion::ROOK;
            } else {

            }

            plane_index = mapper[promotion_type][1 - direction];
        } else if (tolower(this->chessRules.squares[move.src]) == 'k'){
            // get the correct knight move
            direction = Mapper::getKnightDirection(move.src, move.dst);
            plane_index = mapper[KnightMove::NORTH_LEFT + direction][0];
        } else {
            // get the correct direction
            std::tuple<int, int> tuple = Mapper::getQueenDirection(move.src, move.dst);
            plane_index = mapper[std::get<0>(tuple)][std::get<1>(tuple)];
        }
        int row = move.src / 8;
        int col = move.src % 8;
        // create moveProb obj
        moves[move] = outputProbs[plane_index].board[row][col];
    }

    return moves;
}