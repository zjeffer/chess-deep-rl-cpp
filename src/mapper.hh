#pragma once

#include <map>
#include <tuple>
#include <array>


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

extern std::map<int, std::array<int, 8>> mapper;
extern std::array<int, 8> knight_mappings;

class Mapper {
	public:
		static int getUnderpromotionDirection(int src, int dst);
		static int getKnightDirection(int src, int dst);
		static void getQueenDirection(int src, int dst, std::pair<int, int>& mapping);

};

