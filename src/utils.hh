#ifndef UTILS_HH
#define UTILS_HH

#include "chess/thc.hh"
#include <torch/csrc/api/include/torch/torch.h>
#include <torch/torch.h>

void addboardToPlanes(torch::Tensor* planes, int start_index, thc::ChessRules* board) {
    // add the current board to the tensor
    std::cout << "Adding board " << start_index <<  " to planes" << std::endl;
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            switch (board->squares[i * 8 + j]) {
                case ' ': {
                    break;
                }
                case 'P': {
                    planes[0][start_index * 14 + 0][i][j] = 1;
                    break;
                }
                case 'N': {
                    planes[0][start_index * 14 + 1][i][j] = 1;
                    break;
                }
                case 'B': {
                    planes[0][start_index * 14 + 2][i][j] = 1;
                    break;
                }
                case 'R': {
                    planes[0][start_index * 14 + 3][i][j] = 1;
                    break;
                }
                case 'Q': {
                    planes[0][start_index * 14 + 4][i][j] = 1;
                    break;
                }
                case 'K': {
                    planes[0][start_index * 14 + 5][i][j] = 1;
                    break;
                }
                case 'p': {
                    planes[0][start_index * 14 + 6][i][j] = 1;
                    break;
                }
                case 'n': {
                    planes[0][start_index * 14 + 7][i][j] = 1;
                    break;
                }
                case 'b': {
                    planes[0][start_index * 14 + 8][i][j] = 1;
                    break;
                }
                case 'r': {
                    planes[0][start_index * 14 + 9][i][j] = 1;
                    break;
                }
                case 'q': {
                    planes[0][start_index * 14 + 10][i][j] = 1;
                    break;
                }
                case 'k': {
                    planes[0][start_index * 14 + 11][i][j] = 1;
                    break;
                }
                default: {
                    std::cerr << "Error: Invalid piece type: " << board->squares[i * 8 + j] << " at " << i << "," << j << std::endl;
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    int repititions = board->GetRepetitionCount();
    if (repititions >= 1) {
        planes[0][start_index * 14 + 12] = torch::ones({8, 8});
    }
    if (repititions >= 2) {
        planes[0][start_index * 14 + 13] = torch::ones({8, 8});
    }
}


#endif // UTILS_HH