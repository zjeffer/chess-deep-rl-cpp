#include "mapper.hh"
#include <iostream>


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

int Mapper::getUnderpromotionDirection(int src, int dst) {
    int direction = 0;
    int diff = src - dst;
    if (dst < 8) {
        // white promotes (8th rank)
        direction = diff - 8;
    } else if (dst > 55) {
        // white promotes (8th rank)
        direction = diff + 8;
    } else {
        perror("Underpromotion: destination square not on last or first rank");
		exit(EXIT_FAILURE);
    }
    return direction;
}

int Mapper::getKnightDirection(int src, int dst) {
    int direction = -1;
    int diff = src - dst;
    for (int i = 0; i < 8; i++) {
        if (knight_mappings[i] == diff) {
            direction = i;
            break;
        }
    }
    if (direction == -1) {
        perror("Knight direction: not in list?");
		exit(EXIT_FAILURE);
    }
    return direction;
}

void Mapper::getQueenDirection(int src, int dst, std::pair<int, int>& mapping) {
	// TODO: fix for board with square 0 in top left corner
    int diff = src - dst;
    int direction = 0;
    int distance = 0;
    if (diff % 8 == 0) {
        // north and south
        if (diff > 0) {
            direction = QueenDirection::SOUTH;
        } else {
            direction = QueenDirection::NORTH;
        }
        distance = abs(int(diff / 8));
    } else if (diff % 9 == 0) {
        // southwest and northeast
        if (diff > 0) {
            direction = QueenDirection::SOUTHWEST;
        } else {
            direction = QueenDirection::NORTHEAST;
        }
        distance = abs(int(diff / 8));
    } else if (src / 8 == dst / 8) {
        // east and west
        if (diff > 0) {
            direction = QueenDirection::WEST;
        } else {
            direction = QueenDirection::EAST;
        }
        distance = abs(diff);
    } else if (diff % 7 == 0) {
        if (diff > 0) {
            direction = QueenDirection::SOUTHEAST;
        } else {
            direction = QueenDirection::NORTHWEST;
        }
        distance = abs(int(diff / 8)) + 1;
    } else {
        perror("Invalid Queen-like move");
		exit(EXIT_FAILURE);
    }
    mapping.first = direction;
    mapping.second = distance;
}