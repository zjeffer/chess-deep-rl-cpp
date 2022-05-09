#ifndef UTILS_HH
#define UTILS_HH

#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/torch.h>
#include <tuple>

#include "chess/thc.hh"
#include "types.hh"
#include "mapper.hh"

namespace utils {

void addboardToPlanes(torch::Tensor *planes, int start_index, thc::ChessRules *board);

cv::Mat tensorToMat(torch::Tensor tensor, int rows, int cols);

void saveCvMatToImg(const cv::Mat mat, const std::string &filename, int multiplier = 255);

bool isKnightMove(thc::Move move);

std::tuple<int, int, int> moveToPlaneIndex(thc::Move move);

std::map<thc::Move, float> outputProbsToMoves(torch::Tensor &outputProbs, std::vector<thc::Move> legalMoves);

torch::Tensor movesToOutputProbs(std::vector<MoveProb> moves);

bool createDirectory(std::string path);

} // namespace utils

#endif // UTILS_HH