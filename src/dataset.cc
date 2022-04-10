#include "dataset.hh"
#include <filesystem>
#include <iostream>
#include <fstream>

ChessDataSet::ChessDataSet(std::string path) {
	this->createDataset(path);
}

void ChessDataSet::createDataset(std::string path){
	for (auto& p : std::filesystem::directory_iterator(path)) {
		std::string filename = p.path().string();
		std::cout << "Checking file " << filename << " ... ";
		if (filename.find("game-") != std::string::npos) {
			this->addDataFromFile(filename);
			std::cout << "Added data from" << std::endl;
			// std::cout << "Size so far: " << this->data.size() << std::endl;
		} else {
			std::cout << "Skipping..." << std::endl;
		}
	}
}

void ChessDataSet::addDataFromFile(std::string filename) {
	std::cout << "Adding data from file: " << filename << std::endl;
	struct ChessData chessData;
	
	std::ifstream file_stream(filename);
	while(file_stream.read((char*)&chessData, sizeof(struct ChessData))) {
		// printf("%d %d %f\n", chessData.input.size(), chessData.policy.size(), chessData.value);
		// std::cout << this->data.size() << std::endl;

		struct ChessData new_data;
		new_data.input = chessData.input;
		new_data.policy = chessData.policy;
		new_data.value = chessData.value;

		this->data.push_back(new_data);
	}
	file_stream.close();
	std::cout << "Closed file" << std::endl;
}

void ChessDataSet::write(std::string filename, std::vector<ChessData> data) {
	// write to memory/ folder
	try {
		std::filesystem::create_directory("memory");
	} catch (std::filesystem::filesystem_error& e) {
		std::cerr << "Error creating directory: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}

	try {
		std::ofstream file_stream("memory/" + filename, std::ios::out | std::ios::binary);
		for (auto& d : data) {
			d.data = d.data.to(torch::kCPU); // TODO
			file_stream.write((char*)&d, sizeof(struct ChessData));
		}
		file_stream.close();
	} catch (std::filesystem::filesystem_error& e) {
		std::cerr << "Error writing to file: " << e.what() << std::endl;
		exit(EXIT_FAILURE);
	}
}

torch::data::Example<> ChessDataSet::get(size_t index) {
	struct ChessData chessData = data[index];
	return {
		torch::from_blob(chessData.input.data(), {1, 19, 8, 8}), 
		torch::cat({
			torch::from_blob(chessData.policy.data(), {1, 73, 8, 8}), 
			torch::tensor(chessData.value)
		}, 1)};
}

torch::optional<size_t> ChessDataSet::size() const {
	return this->data.size();
}

