#include "train.hh"

Trainer::Trainer(const std::string& networkPath) {
	this->nn = new NeuralNetwork(networkPath);
}

Trainer::~Trainer(){
	std::cout << "Deleting trainer object" << std::endl;
	delete this->nn;
}

void Trainer::train() {
	int batch_size = 64;
	float learning_rate = 0.002;

	// loading the dataset
	ChessDataSet chessDataSet = ChessDataSet("memory");
	if (!g_running) return;
	auto train_set = chessDataSet.map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), batch_size);

	// optimizer
	torch::optim::Adam optimizer(this->nn->getNetwork()->parameters(), learning_rate);
    optimizer.zero_grad();
	
	nn->trainBatches(*data_loader, optimizer, train_set_size, batch_size);
}
