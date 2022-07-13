#pragma once

#include <string_view>

#include "game.hh"
#include "common.hh"

namespace SelfPlay {

int testThread();

int playGame(int amount_of_sims, Agent* white, Agent* black);

void playContinuously(std::string networkPath, int amount_of_sims, int parallel_games);

void playPosition(std::string fen, int amount_of_sims);

} // namespace SelfPlay
