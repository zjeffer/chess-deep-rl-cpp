#include "utils.hh"

void utils::addboardToPlanes(torch::Tensor *planes, int start_index, thc::ChessRules *board) {
    // add the current board to the tensor
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
                std::cerr << board->ToDebugStr() << std::endl;
                exit(EXIT_FAILURE);
            }
            }
        }
    }

    int repetitions = board->GetRepetitionCount();
    if (repetitions > 1) {
        planes[0][start_index * 14 + 12] = torch::ones({8, 8});
    }
    if (repetitions > 2) {
        planes[0][start_index * 14 + 13] = torch::ones({8, 8});
    }
}

cv::Mat utils::tensorToMat(const torch::Tensor &tensor, int rows, int cols) {
    // reshape tensor from 3d tensor to 2d rectangle
    torch::Tensor reshaped = torch::stack(torch::unbind(tensor, 2), 1);

    cv::Mat mat(
        cv::Size{rows, cols},
        CV_32FC1,
        reshaped.data_ptr<float>());
    // without .clone() it will create some weird errors
    return mat.clone();
}

void utils::saveCvMatToImg(const cv::Mat mat, const std::string &filename) {
    // multiply every pixel by 128
    // not 255, so the difference between bools and real values (>1) is clearer
    mat.convertTo(mat, CV_32FC1, 128);
    cv::imwrite(filename, mat);
    std::cout << "Saved image to: " << filename << std::endl;
}


bool utils::isKnightMove(thc::Move move) {
    int src_row = move.src / 8;
    int src_col = move.src % 8;
    int dst_row = move.dst / 8;
    int dst_col = move.dst % 8;

    int row_diff = abs(src_row - dst_row);
    int col_diff = abs(src_col - dst_col);

    return (row_diff == 2 && col_diff == 1) || (row_diff == 1 && col_diff == 2);
}


std::tuple<int, int, int> utils::moveToPlaneIndex(thc::Move move){
    int plane_index = -1;
    int direction = 0;

    if (move.special >= thc::SPECIAL_PROMOTION_ROOK and
        move.special <= thc::SPECIAL_PROMOTION_KNIGHT) {
        // get directions
        direction = Mapper::getUnderpromotionDirection(move.src, move.dst);

        // get type of special move
        int promotion_type;
        if (move.special == thc::SPECIAL_PROMOTION_KNIGHT) {
            promotion_type = UnderPromotion::KNIGHT;
        } else if (move.special == thc::SPECIAL_PROMOTION_BISHOP) {
            promotion_type = UnderPromotion::BISHOP;
        } else if (move.special == thc::SPECIAL_PROMOTION_ROOK) {
            promotion_type = UnderPromotion::ROOK;
        } else {
            printf("Unhandled promotion type: %d\n", move.special);
        }

        plane_index = mapper[promotion_type][1 - direction];
    } else if (utils::isKnightMove(move)) {
        // get the correct knight move
        direction = Mapper::getKnightDirection(move.src, move.dst);
        plane_index = mapper[KnightMove::NORTH_LEFT + direction][0];
    } else {
        // get the correct direction
        std::tuple<int, int> tuple =
            Mapper::getQueenDirection(move.src, move.dst);
        plane_index = mapper[std::get<0>(tuple)][std::get<1>(tuple)];
    }

    if (plane_index < 0 or plane_index > 72) {
        printf("Plane index: %d\n", plane_index);
        perror("Plane index out of bounds!");
        exit(EXIT_FAILURE);
    }

    int row = move.src / 8;
    int col = move.src % 8;
    return std::make_tuple(plane_index, row, col);
}

std::map<thc::Move, float> utils::outputProbsToMoves(std::array<floatBoard, 73> &outputProbs, std::vector<thc::Move> legalMoves) {
    std::map<thc::Move, float> moves = {};

    for (int i = 0; i < (int)legalMoves.size(); i++) {
        std::tuple<int, int, int> tpl = utils::moveToPlaneIndex(legalMoves[i]);
        moves[legalMoves[i]] = outputProbs[std::get<0>(tpl)].board[std::get<1>(tpl)][std::get<2>(tpl)];
    }
    return moves;
}


std::array<floatBoard, 73> utils::movesToOutputProbs(std::vector<MoveProb> moves){
    std::array<floatBoard, 73> output;
    for (MoveProb move : moves){
        std::tuple<int, int, int> tpl = utils::moveToPlaneIndex(move.move);
        output[std::get<0>(tpl)].board[std::get<1>(tpl)][std::get<2>(tpl)] = move.prob;
    }
    return output;
}

bool utils::createDirectory(std::string path){
    return std::filesystem::create_directories(path);
}