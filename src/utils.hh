#pragma once

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#undef slots
#include <torch/torch.h>
#include <torch/jit.h>
#include <torch/nn.h>
#include <torch/script.h>
#define slots Q_SLOTS

#include <tuple>
#include <vector>

#include "chess/thc.hh"
#include "types.hh"
#include "mapper.hh"
#include "environment.hh"
#include "mcts.hh"
#include "utils.hh"
#include "common.hh"

namespace utils {

void addboardToPlanes(torch::Tensor *planes, int start_index, thc::ChessRules *board);

cv::Mat tensorToMat(torch::Tensor tensor, int rows, int cols);

void saveCvMatToImg(const cv::Mat mat, const std::string &filename, int multiplier = 255);

bool isKnightMove(thc::Move move);

std::tuple<int, int, int> moveToPlaneIndex(thc::Move move);

std::map<thc::Move, float> outputProbsToMoves(torch::Tensor &outputProbs, std::vector<thc::Move> legalMoves);

torch::Tensor movesToOutputProbs(std::vector<MoveProb> moves);

std::vector<float> sampleFromGamma(int size);

void addDirichletNoise(Node* root);

std::string getDirectoryFromFilename(std::string filename);

bool createDirectory(std::string path);

void viewTensorFromFile(std::string filename);

std::string getTimeString();

void writeLossToCSV(std::string filename, LossHistory &lossHistory);

/* Tests */

void test_Dirichlet();

void test_MCTS();

void test_NN(std::string networkPath);

void test_Train(std::string networkPath);

void testBug();

} // namespace utils

