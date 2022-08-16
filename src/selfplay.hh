#pragma once

#include <string_view>

#include "game.hh"
#include "common.hh"

namespace SelfPlay {

int playGame(int amount_of_sims, Agent* white, Agent* black);

void playContinuously(const std::shared_ptr<NeuralNetwork> &nn, int amount_of_sims);

void playPosition(const std::shared_ptr<NeuralNetwork> &nn, std::string fen, int amount_of_sims);

} // namespace SelfPlay
