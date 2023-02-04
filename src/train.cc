#include "train.hh"

Trainer::Trainer(const std::string &networkPath, int batch_size,
				 float learning_rate)
	: m_NN(std::make_shared<NeuralNetwork>(networkPath)),
	  m_BatchSize(batch_size), m_LearningRate(learning_rate) {

	  }

Trainer::~Trainer(){
	// std::cout << "Deleting trainer object" << std::endl;
}

void Trainer::train() {
	// loading the dataset
	ChessDataSet chessDataSet = ChessDataSet("memory");
	if (!g_Running) return;
	auto train_set = chessDataSet.map(torch::data::transforms::Stack<>());
	int train_set_size = train_set.size().value();
	auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(train_set), m_BatchSize);

	// optimizer
	torch::optim::Adam optimizer(m_NN->getNetwork()->parameters(), m_LearningRate);
    optimizer.zero_grad();
	
	m_NN->trainBatches(*data_loader, optimizer, train_set_size, m_BatchSize);
}
