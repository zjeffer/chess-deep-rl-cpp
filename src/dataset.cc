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
		if (std::filesystem::is_directory(game.path()) && game.path().string().find(path + "/game-") == 0) {
			// for every move in the folder:
			for (const auto& move : std::filesystem::directory_iterator(game.path())){
				if (!std::filesystem::is_regular_file(move.path())){
					continue;
				}
				torch::Tensor input, outputs;
				try {
					if (move.path().string().find("input.pt") != std::string::npos) {
						torch::load(input, move.path());
						// also load corresponding output
						std::string output_path = move.path().string().replace(move.path().string().find("input.pt"), 8, "output.pt");
						torch::load(outputs, output_path);
					} 
				} catch (std::exception& e) {
					std::cerr << "Could not load tensor from file: " << move.path() << std::endl;
					std::cerr << e.what() << std::endl;
					continue;
				}
				this->data.push_back(std::make_pair(input, outputs));
			}
		} else {
			std::cout << "Skipping " << game.path().string() << std::endl;
		}
	}
	std::cout << "Created dataset of size " << this->data.size() << std::endl;
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	torch::Tensor input = this->data[index].first;
	torch::Tensor output = this->data[index].second;
	return {input, output};
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

