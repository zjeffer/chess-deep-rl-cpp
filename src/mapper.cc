#include "mapper.hh"

#include <iostream>

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
		exit(1);
    }
    return direction;
}

int Mapper::getKnightDirection(int src, int dst) {
    int direction = NULL;
    int diff = src - dst;
    for (int i = 0; i < 8; i++) {
        if (knight_mappings[i] == diff) {
            direction = i;
            break;
        }
    }
    if (direction == NULL) {
        perror("Knight direction: not in list?");
		exit(1);
    }
    return direction;
}

std::tuple<int, int> Mapper::getQueenDirection(int src, int dst) {
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
        distance = int(diff / 8);
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
		exit(1);
    }
    return std::make_tuple(direction, distance);
}