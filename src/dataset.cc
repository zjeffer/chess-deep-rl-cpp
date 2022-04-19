#include "dataset.hh"
#include <filesystem>
#include <iostream>
#include <fstream>

ChessDataSet::ChessDataSet(std::string path) {
	this->createDataset(path);
}

void ChessDataSet::createDataset(std::string path){
	// for every folder in path/
	for (const auto& game : std::filesystem::directory_iterator(path)) {
		// if folder is a folder and starts with game-
		if (std::filesystem::is_directory(game.path()) && game.path().string().find("game-") == 0) {
			torch::Tensor inputs;
			torch::load(inputs, game.path().string() + "/input.pt");
			torch::Tensor outputs;
			torch::load(outputs, game.path().string() + "/outputs.pt");
			// save tensors to data
			
		}
	}
	std::cout << "Created dataset" << std::endl;
}

void ChessDataSet::read(std::string filename, torch::Tensor data) {
	torch::load(data, filename);
}

void ChessDataSet::write(std::string filename, torch::Tensor data) {
	torch::save(data, filename);
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	return this->data[index];
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

