#include "move.hh"

Move::Move(int from_square, int to_square, int promotion_piece){
	this->from_square = from_square;
	this->to_square = to_square;
	this->promotion_piece = promotion_piece;
}

Move::~Move(){
	
}

