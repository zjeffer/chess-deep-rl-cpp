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
                LOG(FATAL) << "Error: Invalid piece type: " << board->squares[i * 8 + j] << " at " << i << "," << j << "\n"
                << board->ToDebugStr();
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

cv::Mat utils::tensorToMat(torch::Tensor tensor, int rows, int cols) {
    // if tensor is 3d, add a dimension at the start
    if (tensor.dim() == 3) {
        tensor = tensor.unsqueeze(0);
    }
    // reshape tensor from 3d tensor to 2d rectangle
    torch::Tensor reshaped = torch::stack(torch::unbind(tensor, 2), 1).flatten(0, 1);
    LOG(DEBUG) << "Reshaped tensor from " << tensor.sizes() << " to " << reshaped.sizes();

    // send to cpu, otherwise it segfaults
    reshaped = reshaped.to(torch::kCPU);
    cv::Mat mat(
        cv::Size{rows, cols},
        CV_32FC1,
        reshaped.data_ptr<float>());
    // without .clone() it will create some weird errors
    return mat.clone();
}

void utils::saveCvMatToImg(const cv::Mat mat, const std::string &filename, int multiplier) {
    // multiply every pixel by a multiplier
    // this is because CV expects values from 0-255
    LOG(DEBUG) << "Converting mat...";
    LOG(DEBUG) << "Multiplier: " << multiplier;
    mat.convertTo(mat, CV_32FC1, multiplier * 255);
    if (cv::imwrite(filename, mat)) {
        LOG(DEBUG) << "Saved image to: " << filename;
    } else {
        LOG(WARNING) << "Error: Could not save image to: " << filename;
    }
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
        std::pair<int, int> mapping;
        Mapper::getQueenDirection(move.src, move.dst, mapping);
        plane_index = mapper[mapping.first][mapping.second];
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

std::map<thc::Move, float> utils::outputProbsToMoves(torch::Tensor &outputProbs, std::vector<thc::Move> legalMoves) {
    std::map<thc::Move, float> moves = {};

    for (int i = 0; i < (int)legalMoves.size(); i++) {
        std::tuple<int, int, int> tpl = utils::moveToPlaneIndex(legalMoves[i]);
        moves[legalMoves[i]] = outputProbs[std::get<0>(tpl)][std::get<1>(tpl)][std::get<2>(tpl)].item().toFloat();
    }
    return moves;
}


torch::Tensor utils::movesToOutputProbs(std::vector<MoveProb> moves){
    torch::Tensor output = torch::zeros({1, 73, 8, 8});
    for (MoveProb move : moves){
        std::tuple<int, int, int> tpl = utils::moveToPlaneIndex(move.move);
        output[std::get<0>(tpl)][std::get<1>(tpl)][std::get<2>(tpl)] = move.prob;
    }
    return output;
}

std::vector<float> utils::sampleFromGamma(int size){
    std::vector<float> samples = std::vector<float>(size, 0.0f);
    float sum = 0.0f;
    for (int i = 0; i < size; i++){
        samples[i] = g_distribution(g_Generator);
        sum += samples[i];
    }
    for (int i = 0; i < size; i++){
        samples[i] /= sum;
    }
    return samples;
}

void utils::addDirichletNoise(Node* root) {
    std::vector<Node*> children = root->getChildren();

    // get the noise
    std::vector<float> noise = sampleFromGamma(children.size());

    float frac = 0.25;

    for (int i = 0; i < (int)children.size(); i++) {
        children[i]->setPrior(children[i]->getPrior() * (1 - frac) + noise[i] * frac);
    }
}

std::string utils::getDirectoryFromFilename(std::string filename){
    if (filename.find_last_of("/") == std::string::npos) {
        return "";
    }
    return filename.substr(0, filename.find_last_of("/"));
}

bool utils::createDirectory(std::string path){
    return std::filesystem::create_directories(path);
}

void utils::writeLossToCSV(std::string filename, LossHistory &lossHistory){
    if (!filename.ends_with(".csv")) {
        filename += ".csv";
    }
    // check if filename is in subdirectory
    std::string dir;
    if (!(dir = utils::getDirectoryFromFilename(filename)).empty()) {
        // if it is, create the directory to be sure
        createDirectory(dir);
    }
    LOG(INFO) << "Saving loss history to " << filename;
    std::ofstream file;
    file.open(filename);
    // header
    file << "Epoch;Policy Loss;Value Loss;Total Loss" << std::endl;
    for (int i = 0; i < lossHistory.historySize; i++) {
        file << i << ";" << lossHistory.policies[i] << ";" << lossHistory.values[i] << ";" << lossHistory.losses[i] << "\n";
    }
    file << "\n";
    file.close();
}

void utils::test_Dirichlet(){
    LOG(INFO) << "Testing Dirichlet...";
    std::shared_ptr<NeuralNetwork> nn = std::make_shared<NeuralNetwork>();

    MCTS mcts = MCTS(new Node(), nn);
    mcts.run_simulations(400);
}

void utils::test_MCTS(){
	Environment env = Environment();
	LOG(DEBUG) << env.getFen();

	// test mcts tree
    std::shared_ptr<NeuralNetwork> network = std::make_shared<NeuralNetwork>();
	MCTS mcts = MCTS(new Node(), network);

	// run sims
	mcts.run_simulations(400);

	// show actions of root
	Node* root = mcts.getRoot();
	std::vector<Node*> nodes = root->getChildren();
	thc::ChessRules* cr = env.getRules();
	printf("Possible moves in state %s: \n", env.getFen().c_str());
	for (int i = 0; i < (int)nodes.size(); i++) {
		printf("%s \t Prior: %f \t Q: %f \t U: %f\n", nodes[i]->getAction().NaturalOut(cr).c_str(), nodes[i]->getPrior(), nodes[i]->getQ(), nodes[i]->getUCB());
	}
}

void utils::test_NN(std::string networkPath){
	NeuralNetwork nn = NeuralNetwork(networkPath);
    Environment board = Environment();

	// test board to input
	LOG(DEBUG) << "Converting board to input state";
	torch::Tensor input = board.boardToInput();

	// save input to image
	// LOG(DEBUG) << "Converting input to image";
	// cv::Mat mat = utils::tensorToMat(input.clone(), 119*8, 8);
	// utils::saveCvMatToImg(mat, "tests/input.png", 128);

	// predict
    std::tuple<torch::Tensor, torch::Tensor> outputs = nn.predict(input);
    torch::Tensor policyOutput = std::get<0>(outputs).view({73, 8, 8});
    float valueOutput = std::get<1>(outputs).item<float>();
    if (!policyOutput.nan_to_num().equal(policyOutput)){
        LOG(WARNING) << policyOutput;
        LOG(WARNING) << "Output is NaN";
        exit(EXIT_FAILURE);
    }

	LOG(DEBUG) << "predicted";

	// value is the last element of the output tensor
	LOG(DEBUG) << "policy: " << policyOutput.sizes();
	LOG(DEBUG) << "value: " << valueOutput;

	// save output to img
    LOG(DEBUG) << "Converting output to image";
	cv::Mat img = utils::tensorToMat(policyOutput.clone(), 73*8, 8);
	LOG(DEBUG) << "image: " << img.size();

    // calculate the appropriate multiplier
    float max_value = torch::max(policyOutput).item<float>();
    LOG(DEBUG) << "max_value: " << max_value;
	utils::saveCvMatToImg(img, "tests/output.png", 1.0/max_value);
}

void utils::test_Train(std::string networkPath){
	NeuralNetwork nn = NeuralNetwork(networkPath);

	ChessDataSet chessDataSet = ChessDataSet("memory");
	
	auto train_set = chessDataSet.map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	int batch_size = 256;
	
	// data loader
	LOG(DEBUG) << "Creating data loader";
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);

	LOG(DEBUG) << "Data loader created";


	// optimizer
	float learning_rate = 0.5;
	torch::optim::Adam optimizer(nn.getNetwork()->parameters(), learning_rate);
    optimizer.zero_grad();

	
	nn.trainBatches(*data_loader, optimizer, train_set_size, batch_size);
}

void utils::testBug(){
    std::string fen = "1r1q1b1r/p1pkPp2/7p/1p1p4/3P3Q/Pn2P1P1/1PPB3R/RN1K1BN1 b - - 0 22";
    thc::Move move;
    // create environment
    Environment env = Environment(fen);
    // make rook move up
    if(!move.TerseIn(env.getRules(), "h8h7")){
        LOG(FATAL) << "Invalid move";
        exit(EXIT_FAILURE);
    }
    env.makeMove(move);
    env.printBoard();
}

std::string utils::getTimeString() {
    time_t rawtime;
    struct tm * timeinfo;
    char buffer[80];

    time(&rawtime);
    timeinfo = localtime(&rawtime);

    strftime(buffer, 80, "%Y-%m-%d_%H-%M-%S", timeinfo);
    return std::string(buffer);
}

void utils::viewTensorFromFile(std::string filename) {
    torch::Tensor tensor;
    torch::load(tensor, filename);
    LOG(INFO) << "Tensor: " << tensor;
    LOG(INFO) << "Tensor size: " << tensor.sizes();
}