#pragma once

#include <string_view>

#include "game.hh"
#include "common.hh"

namespace SelfPlay {

int playGame(int amount_of_sims, Agent* white, Agent* black);

void playContinuously(NeuralNetwork* nn, int amount_of_sims, int parallel_games);

void playPosition(std::string fen, int amount_of_sims);

} // namespace SelfPlay
