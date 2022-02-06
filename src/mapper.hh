#ifndef mapper_hh
#define mapper_hh

#include <map>
#include <tuple>


enum QueenDirection  {
	NORTHWEST = 0,
    NORTH = 1,
    NORTHEAST = 2,
    EAST = 3,
    SOUTHEAST = 4,
    SOUTH = 5,
    SOUTHWEST = 6,
    WEST = 7
};


enum KnightMove  {
    // eight possible knight moves
    NORTH_LEFT = 0,  // diff == -15
    NORTH_RIGHT = 1,  // diff == -17
    EAST_UP = 2,  // diff == -6
    EAST_DOWN = 3,  // diff == 10
    SOUTH_RIGHT = 4,  // diff == 15
    SOUTH_LEFT = 5,  // diff == 17
    WEST_DOWN = 6,  // diff == 6
    WEST_UP = 7  // diff == -10
};

enum UnderPromotion  {
    KNIGHT = 0,
    BISHOP = 1,
    ROOK = 2
};


std::map<int, std::array<int, 8>> mapper = {
	// queen-like moves
	{QueenDirection::NORTHWEST, {0, 1, 2, 3, 4, 5, 6}},
	{QueenDirection::NORTH, {7, 8, 9, 10, 11, 12, 13}},
	{QueenDirection::NORTHEAST, {14, 15, 16, 17, 18, 19, 20}},
	{QueenDirection::EAST, {21, 22, 23, 24, 25, 26, 27}},
	{QueenDirection::SOUTHEAST, {28, 29, 30, 31, 32, 33, 34}},
	{QueenDirection::SOUTH, {35, 36, 37, 38, 39, 40, 41}},
	{QueenDirection::SOUTHWEST, {42, 43, 44, 45, 46, 47, 48}},
	{QueenDirection::WEST, {49, 50, 51, 52, 53, 54, 55}},
	// knights
	{KnightMove::NORTH_LEFT, {56}},
	{KnightMove::NORTH_RIGHT, {57}},
	{KnightMove::EAST_UP, {58}},
	{KnightMove::EAST_DOWN, {59}},
	{KnightMove::SOUTH_RIGHT, {60}},
	{KnightMove::SOUTH_LEFT, {61}},
	{KnightMove::WEST_DOWN, {62}},
	{KnightMove::WEST_UP, {63}},
	// underpromotions
	{UnderPromotion::KNIGHT, {64, 65, 66}},
	{UnderPromotion::BISHOP, {67, 68, 69}},
	{UnderPromotion::ROOK, {70, 71, 72}}
};

std::array<int, 8> knight_mappings = {15, 17, 6, -10, -15, -17, -6, 10};

class Mapper {
	public:
		static int getUnderpromotionDirection(int src, int dst);
		static int getKnightDirection(int src, int dst);
		static std::tuple<int, int> getQueenDirection(int src, int dst);

};

#endif // mapper_hh