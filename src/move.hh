#ifndef move_hh
#define move_hh

class Move {
	public:
		Move(int from_square, int to_square, int promotion_piece);
		~Move();

	private:
		int from_square;
		int to_square;
		int promotion_piece;
};

#endif /* move_hh */