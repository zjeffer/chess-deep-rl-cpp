#pragma once

#include <string_view>

#include "game.hh"
#include "common.hh"

class MainWindow;

namespace SelfPlay {

int playGame(int amount_of_sims, Agent* white, Agent* black, MainWindow* mainWindow);

void playContinuously(NeuralNetwork* nn, int amount_of_sims, int parallel_games, MainWindow* mainWindow);

void playPosition(std::string fen, int amount_of_sims, MainWindow* mainWindow);

} // namespace SelfPlay
