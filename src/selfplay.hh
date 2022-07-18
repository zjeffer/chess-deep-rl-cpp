#pragma once

#include <string_view>

#include "game.hh"
#include "common.hh"

class MainWindow;

namespace SelfPlay {

int playGame(int amount_of_sims, Agent* white, Agent* black, MainWindow* mainWindow);

void playContinuously(const std::shared_ptr<NeuralNetwork> &nn, int amount_of_sims, MainWindow* mainWindow);

void playPosition(const std::shared_ptr<NeuralNetwork> &nn, std::string fen, int amount_of_sims, MainWindow* mainWindow);

} // namespace SelfPlay
