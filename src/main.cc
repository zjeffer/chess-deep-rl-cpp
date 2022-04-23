#include <iostream>
#include <opencv2/opencv.hpp>

#include "environment.hh"
#include "mcts.hh"
#include "game.hh"
#include "neuralnet.hh"
#include "utils.hh"


void test_MCTS(){
	Environment env = Environment();
	std::cout << env.getFen() << std::endl;

	// test mcts tree
	MCTS mcts = MCTS(new Node(), new NeuralNetwork());

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

void test_NN(){
	NeuralNetwork nn = NeuralNetwork(true);

	Environment board = Environment();
	std::vector<std::string> moveList = {
		"e2e4", 
		"e7e5",
		"g1f3",
		"b8c6",
		"f1c4",
		"f8c5",
		"e1g1"
	};

	// play the moves
	for (std::string moveString : moveList){
		thc::Move move;
		if (move.TerseIn(board.getRules(), moveString.c_str())){
			board.makeMove(move);
		} else {
			std::cerr << "Invalid move: " << moveString << std::endl;
			exit(EXIT_FAILURE);
		}
	}

	// test board to input
	std::cout << "Converting board to input state" << std::endl;
	torch::Tensor input = board.boardToInput();

	// tensor to image
	std::cout << "Converting input to image" << std::endl;
	cv::Mat mat = utils::tensorToMat(input, 119*8, 8);
	utils::saveCvMatToImg(mat, "tests/input.png");

	torch::Tensor output = torch::zeros({4673});

	// predict
	nn.predict(input, output);

	std::cout << "predicted" << std::endl;

	// value is the last element of the output tensor
	torch::Tensor value = output.slice(1, 4672, 4673);
	std::cout << "value: " << value << std::endl;
	torch::Tensor policy = output.slice(1, 0, 4672).view({73, 8, 8});
	std::cout << "policy: " << policy.sizes() << std::endl;
	// reshape to 73x8x8
	cv::Mat img = utils::tensorToMat(policy, 73*8, 8);
	utils::saveCvMatToImg(img, "tests/output.png");
}

void test_Train(){
	NeuralNetwork nn = NeuralNetwork();

	auto train_set = ChessDataSet("memory").map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	int batch_size = 64;
	
	// data loader
	std::cout << "Creating data loader" << std::endl;
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(std::move(train_set), 8);
	std::cout << "Data loader created" << std::endl;

	// optimizer
	int learning_rate = 0.2;
	torch::optim::Adam optimizer(nn.parameters(), learning_rate);

	float Loss = 0, Acc = 0;

	// TODO: fix "stack expects each tensor to be equal size, but got [0] at entry 0 and [119, 8, 8] at entry 2"
	std::cout << "Starting training with " << train_set_size << " examples" << std::endl;
	for (auto& batch: *data_loader){
		std::cout << "Batch" << std::endl;
		auto data = batch.data.to(torch::kCUDA);
		auto target = batch.target.to(torch::kCUDA);
		// divide policy and value targets
		auto policy_target = target.slice(1, 0, 4672).view({73, 8, 8});
		auto value_target = target.slice(1, 4672, 4673);

		auto output = nn.forward(data);
		auto policy_output = output.slice(1, 0, 4672).view({73, 8, 8});
		auto value_output = output.slice(1, 4672, 4673);

		// loss
		// policy loss is categorical cross entropy
		auto policy_loss = -torch::sum(policy_output * torch::log_softmax(policy_output, 1), 1);
		auto value_loss = torch::mse_loss(value_output, value_target);
		
		auto loss = policy_loss + value_loss;
		std::cout << "Loss: " << loss << std::endl;
		loss.backward();
		optimizer.step();


		Loss += loss.item<float>();
		Acc += torch::sum(torch::argmax(policy_output, 1) == torch::argmax(policy_target, 1)).item<float>();
		std::cout << "Loss: " << Loss << " Acc: " << Acc << std::endl;
	}
	std::cout << "Training finished" << std::endl;
	
	// nn.train(*data_loader, optimizer, train_set_size);
}

int playGame(int argc, char** argv){
	int amount_of_sims = 20;
	if (argc == 2) {
		try {
			amount_of_sims = std::stoi(argv[1]);
		} catch (std::invalid_argument) {
			std::cerr << "Invalid argument" << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	// create the NN
	NeuralNetwork* nn = new NeuralNetwork();
	// create the agents, both using the same NN for selfplay
	Agent white = Agent("white", nn);
	Agent black = Agent("black", nn);

	// play one game
	Game game = Game(amount_of_sims, Environment(), white, black);
	return game.playGame();
}

int main(int argc, char** argv) {

	// test mcts simulations:
	// test_MCTS();

	// test neural network input & outputs:
	// test_NN();

	// try training
	test_Train();

	// play chess
	// int winner = playGame(argc, argv);


	return 0;
}