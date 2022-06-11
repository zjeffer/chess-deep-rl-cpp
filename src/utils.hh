#ifndef UTILS_HH
#define UTILS_HH

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/torch.h>
#include <tuple>
#include <vector>

#include "chess/thc.hh"
#include "types.hh"
#include "mapper.hh"
#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
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

std::vector<float> sampleFromGamma(float alpha, float scale, int size);

void addDirichletNoise(Node* root, float alpha=0.3);

bool createDirectory(std::string path);

void test_Dirichlet();

void test_MCTS();

void test_NN(std::string networkPath);

void test_Train();

void testBug();

} // namespace utils

#endif // UTILS_HH