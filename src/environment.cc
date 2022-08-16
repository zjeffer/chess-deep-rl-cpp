#include "environment.hh"
#include <string>
#include <iostream>
#include "utils.hh"


Environment::Environment(thc::ChessRules rules) {
	m_ChessRules = rules;
	this->terminalState = thc::NOT_TERMINAL;
}

Environment::Environment(std::string fen): Environment(thc::ChessRules()) {
    m_ChessRules.Forsyth(fen.c_str());
    m_Fen = m_ChessRules.ForsythPublish();
}

Environment::Environment() : Environment(thc::ChessRules()) {
	
}

thc::ChessRules* Environment::getRules() {
	return &(m_ChessRules);
}

bool Environment::isGameOver() {
	// TODO: implement
	bool okay = m_ChessRules.Evaluate(this->terminalState);

	if (!okay){
		LOG(FATAL) << "Error: Game received illegal position" << std::endl;
		exit(EXIT_FAILURE);
	}

	return this->terminalState != thc::TERMINAL::NOT_TERMINAL;
}

void Environment::reset() {
	if (!m_ChessRules.Forsyth("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")){
        LOG(FATAL) << "Error: Could not reset environment";
        exit(EXIT_FAILURE);
    }
}

void Environment::printDrawType(thc::DRAWTYPE drawType) {
    switch (drawType) {
        case thc::DRAWTYPE::DRAWTYPE_50MOVE:
            LOG(INFO) << "Draw by 50 move rule";
            break;
        case thc::DRAWTYPE::DRAWTYPE_INSUFFICIENT:
            LOG(INFO) << "Draw by insufficient material";
            break;
        case thc::DRAWTYPE::DRAWTYPE_INSUFFICIENT_AUTO:
            LOG(INFO) << "Draw by insufficient material (auto)";
            break;
        case thc::DRAWTYPE::DRAWTYPE_REPITITION:
            LOG(INFO) << "Draw by repetition";
            break;
        case thc::DRAWTYPE::NOT_DRAW:
            LOG(INFO) << "Not a draw";
            break;
        default:
            LOG(WARNING) << "Unknown draw type";
            break;
    }
}

void Environment::printBoard() {
    LOG(INFO) << "\n" << m_ChessRules.ToDebugStr();
}

bool Environment::getCurrentPlayer() const {
	return m_ChessRules.WhiteToPlay();
}

const std::string& Environment::getFen() {
	m_Fen = m_ChessRules.ForsythPublish();
    return m_Fen;
}

int Environment::getAmountOfPieces(){
    int pieces = 0;
    for (int i = 0; i < 64; i++){
        if (m_ChessRules.squares[i] != ' '){
            pieces++;
        }
    }
    return pieces;
}

std::string Environment::makeMove(thc::Move move) {
    // check if move is empty
    if (!move.Valid()){
        LOG(FATAL) << "Move is not valid" << std::endl;
        exit(EXIT_FAILURE);
    }
    int amountOfPiecesBeforeMove = this->getAmountOfPieces();
	m_ChessRules.PlayMove(move);
    int amountOfPiecesAfterMove = this->getAmountOfPieces();
    if (abs(amountOfPiecesBeforeMove - amountOfPiecesAfterMove) > 1){
        LOG(FATAL) << "Move did not result in correct amount of pieces" << std::endl;
        exit(EXIT_FAILURE);
    }
    return this->getFen();
}

std::string Environment::pushMove(thc::Move move) {
    m_ChessRules.PushMove(move);
    return this->getFen();
}

std::string Environment::undoMove(thc::Move move) {
    m_ChessRules.PopMove(move);
    return this->getFen();
}

void Environment::getLegalMoves(std::vector<thc::Move> &moves) {
    m_ChessRules.GenLegalMoveList(moves);
}

torch::Tensor Environment::boardToInput() {
    torch::Tensor input = torch::zeros({119, 8, 8});

    // add all the pieces and the repitition counts (14 planes * 8 moves)
    thc::ChessRules currentBoard (m_ChessRules);
    // TODO: delete below line if it works
    // std::memcpy(&currentBoard, &m_ChessRules, sizeof(thc::ChessRules));
    for (int i = 0; i < (int)m_ChessRules.history_idx; i++) {
        utils::addboardToPlanes(&input, i, &currentBoard);
        currentBoard.PopMove(currentBoard.history[currentBoard.history_idx-i-1]);
    }

    // current turn (plane 113)
    if (m_ChessRules.WhiteToPlay()) {
        input[14*8] = torch::ones({8, 8});
    } else {
        input[14*8] = torch::zeros({8, 8});
    }

    // total move counter (plane 114)
    input[14*8 + 1] = torch::full({8, 8}, m_ChessRules.full_move_count);

    // castling rights (planes 115-118)
    input[14*8 + 2] = torch::full({8, 8}, m_ChessRules.wqueen == 1);
    input[14*8 + 3] = torch::full({8, 8}, m_ChessRules.wking == 1);
    input[14*8 + 4] = torch::full({8, 8}, m_ChessRules.bqueen == 1);
    input[14*8 + 5] = torch::full({8, 8}, m_ChessRules.bking == 1);

    // no-progress counter (plane 119)
    input[14*8 + 6] = torch::full({8, 8}, m_ChessRules.half_move_clock);
    
    // expand dims
    input = input.unsqueeze(0);
    return input;
}



